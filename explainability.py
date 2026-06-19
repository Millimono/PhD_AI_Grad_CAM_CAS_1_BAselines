"""
explainability.py  (batched edition)
------------------------------------
Same four faithfulness metrics — deletion, insertion, comprehensiveness,
sufficiency — but the perturbations of a single image are stacked into one
tensor and forwarded together. On a GPU this is roughly 10-20x faster than
the per-step loop, at the cost of more VRAM per image.

Public API is unchanged:
    compute_metrics_on_loader(model, gradcam, dataloader, ...)

Directions of "better":
    deletion          ↓ lower  is better
    insertion         ↑ higher is better
    comprehensiveness ↑ higher is better
    sufficiency       ↓ lower  is better
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# np.trapz was renamed to np.trapezoid in NumPy 2.0 — support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# --------------------------------------------------------------------------- #
#  Saliency (Grad-CAM) for the whole batch                                    #
# --------------------------------------------------------------------------- #
def compute_cams(model, gradcam, images, labels, device):
    """Returns saliency maps upsampled to the input resolution, normalized to [0,1].
    Shape: (B, 1, H, W), detached."""
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    outputs, features = model(images)
    target = outputs.gather(1, labels.view(-1, 1)).squeeze(1).sum()
    grads = torch.autograd.grad(target, features, retain_graph=False, create_graph=False)[0]

    cam, _, _ = gradcam(features, grads)                                # (B,1,h,w)
    cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)

    B, _, H, W = cam.shape
    flat = cam.view(B, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - mn) / (mx - mn + 1e-8)
    return flat.view(B, 1, H, W).detach()


def _make_baseline(img, kind="black"):
    if kind == "black":
        return torch.zeros_like(img)
    if kind == "blur":
        k = 11
        coords = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * 5.0 ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).expand(img.shape[1], 1, k, k)
        return F.conv2d(img, kernel, padding=k // 2, groups=img.shape[1])
    raise ValueError(f"Unknown baseline '{kind}'.")


# --------------------------------------------------------------------------- #
#  Per-image batched probability traces                                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _probs_along_schedule(model, img, baseline, order, schedule, label, mode):
    """Build one perturbed image per step in `schedule`, forward the whole stack
    in chunks, and return the target-class probability at each step.

    mode = "delete": start from `img`, progressively replace top-k with baseline.
    mode = "insert": start from `baseline`, progressively reveal top-k of `img`.
    """
    H, W = img.shape[-2:]
    n = H * W
    S = len(schedule)

    # Build the (S, 1, 1, H*W) mask stack in one shot.
    # mask[s] keeps pixels that are still "image" at step s.
    masks = torch.ones((S, n), device=img.device) if mode == "delete" \
            else torch.zeros((S, n), device=img.device)
    for s, k in enumerate(schedule):
        if mode == "delete":
            masks[s, order[:k]] = 0.0
        else:
            masks[s, order[:k]] = 1.0
    masks = masks.view(S, 1, H, W)                                      # (S,1,H,W)

    # Broadcast the single image / baseline across S
    img_S = img.expand(S, -1, -1, -1)                                   # (S,C,H,W)
    base_S = baseline.expand(S, -1, -1, -1)
    pert = img_S * masks + base_S * (1 - masks)                         # (S,C,H,W)

    # Chunk the forward pass so very long schedules don't blow VRAM.
    # 64 perturbed copies of a 224x224 image ≈ a normal training batch.
    chunk = 64
    out_probs = []
    for i in range(0, S, chunk):
        logits = model(pert[i:i + chunk])[0]
        out_probs.append(F.softmax(logits, dim=1)[:, label])
    return torch.cat(out_probs).cpu().numpy()                           # (S,)


@torch.no_grad()
def _image_metrics(model, img, cam, label, baseline, steps, keep_fractions):
    """All four metrics for ONE image, with at most 2 batched forwards (one for
    deletion, one for insertion). Comp/suff reuse the deletion/insertion stacks
    by computing them as extra entries in the schedule."""
    H, W = cam.shape[-2:]
    n = H * W
    order = torch.argsort(cam.flatten(), descending=True)

    # ---- Deletion schedule + extra entries for comprehensiveness (remove top-kf%)
    del_schedule = [int(round(s / steps * n)) for s in range(steps + 1)]
    comp_idx0 = len(del_schedule)
    del_schedule += [int(round(kf * n)) for kf in keep_fractions]

    del_probs = _probs_along_schedule(model, img, baseline, order, del_schedule, label, "delete")
    del_curve = del_probs[:steps + 1]
    del_auc = float(_trapz(del_curve, dx=1.0 / steps))
    p_full = float(del_probs[0])                                   # k=0 → unperturbed input
    p_after_remove = del_probs[comp_idx0:]                          # per kf
    comp = float(np.mean(p_full - p_after_remove))

    # ---- Insertion schedule + extra entries for sufficiency (keep only top-kf%)
    ins_schedule = [int(round(s / steps * n)) for s in range(steps + 1)]
    suff_idx0 = len(ins_schedule)
    ins_schedule += [int(round(kf * n)) for kf in keep_fractions]

    ins_probs = _probs_along_schedule(model, img, baseline, order, ins_schedule, label, "insert")
    ins_curve = ins_probs[:steps + 1]
    ins_auc = float(_trapz(ins_curve, dx=1.0 / steps))
    p_kept_only = ins_probs[suff_idx0:]
    suff = float(np.mean(p_full - p_kept_only))

    return {
        "label": int(label),
        "deletion": del_auc,
        "insertion": ins_auc,
        "comprehensiveness": comp,
        "sufficiency": suff,
    }


# --------------------------------------------------------------------------- #
#  Dataset-level driver                                                       #
# --------------------------------------------------------------------------- #
def compute_metrics_on_loader(model, gradcam, dataloader, device="cuda",
                              steps=50, keep_fractions=(0.5,), baseline="black",
                              csv_path=None, progress=True):
    """Runs the four metrics over a DataLoader and returns the means plus a
    per-image record list. Optionally writes a CSV."""
    try:
        from tqdm import tqdm
        iterator = tqdm(dataloader, disable=not progress)
    except ImportError:
        iterator = dataloader

    model.eval()
    records = []
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        cams = compute_cams(model, gradcam, images, labels, device)     # (B,1,H,W)

        for i in range(images.shape[0]):
            img = images[i:i + 1]
            cam = cams[i, 0]
            label = int(labels[i].item())
            base = _make_baseline(img, baseline)
            records.append(_image_metrics(model, img, cam, label, base, steps, keep_fractions))

    means = {k: float(np.mean([r[k] for r in records]))
             for k in ("deletion", "insertion", "comprehensiveness", "sufficiency")}

    if csv_path:
        try:
            import pandas as pd
            pd.DataFrame(records).to_csv(csv_path, index=False)
        except ImportError:
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=records[0].keys())
                w.writeheader()
                w.writerows(records)

    return means, records