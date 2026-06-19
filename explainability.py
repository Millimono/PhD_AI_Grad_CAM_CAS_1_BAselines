"""
explainability.py
-----------------
Faithfulness metrics for Grad-CAM attention maps:
deletion, insertion, sufficiency, comprehensiveness.

Recovered and cleaned from PhD_AI_Grad_CAM_CAS_*.ipynb. The metrics are
model-agnostic: they only need a model returning ``(logits, features)``
(your ``FullModel``) and a Grad-CAM module returning ``(cam, cam_pre_relu,
weights)`` (your ``DifferentiableGradCAM``).

Direction of "better":
    deletion        ↓ lower  is better  (removing salient pixels collapses the prediction fast)
    insertion       ↑ higher is better  (inserting salient pixels recovers the prediction fast)
    comprehensiveness ↑ higher is better  (removing the rationale drops the probability a lot)
    sufficiency     ↓ lower  is better  (keeping only the rationale barely drops the probability)

Note: the two notebook versions disagreed on comprehensiveness/sufficiency
(one returned a probability *drop*, the other a *residual* probability).
This module implements the ERASER-style **drop** convention, which is the
one consistent with the comprehensiveness values reported in the paper
(e.g. up to 0.83 on CIFAR-10).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# np.trapz was renamed to np.trapezoid in NumPy 2.0 — support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# --------------------------------------------------------------------------- #
#  Saliency (Grad-CAM) computation for a whole batch                          #
# --------------------------------------------------------------------------- #
def compute_cams(model, gradcam, images, labels, device):
    """
    Returns per-image saliency maps upsampled to the input resolution,
    normalized to [0, 1]. Shape: (B, 1, H, W), detached.
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    # Grad-CAM needs gradients of the target logit w.r.t. the features.
    # FullModel detaches `features` internally and sets requires_grad, so it is
    # a leaf we can differentiate against.
    outputs, features = model(images)
    target = outputs.gather(1, labels.view(-1, 1)).squeeze(1).sum()
    grads = torch.autograd.grad(target, features, retain_graph=False, create_graph=False)[0]

    cam, _, _ = gradcam(features, grads)          # (B, 1, h, w)
    cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)

    # Per-image min-max normalization
    B = cam.shape[0]
    H, W = images.shape[2:]
    flat = cam.view(B, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - mn) / (mx - mn + 1e-8)
    cam = flat.view(B, 1, H, W)
    return cam.detach()


def _make_baseline(img, kind="black"):
    """Reference image used to 'erase' a region. 'black' = zeros; 'blur' = heavily blurred input."""
    if kind == "black":
        return torch.zeros_like(img)
    if kind == "blur":
        k = 11
        coords = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * 5.0 ** 2))
        g = (g / g.sum())
        kernel = (g[:, None] * g[None, :]).expand(img.shape[1], 1, k, k)
        return F.conv2d(img, kernel, padding=k // 2, groups=img.shape[1])
    raise ValueError(f"Unknown baseline '{kind}' (use 'black' or 'blur').")


# --------------------------------------------------------------------------- #
#  Deletion / Insertion (RISE, Petsiuk et al. 2018)                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def deletion_auc(model, img, cam, label, baseline, steps=50):
    """Remove most-salient pixels first; AUC of the target-class probability. Lower is better."""
    H, W = cam.shape[-2:]
    order = torch.argsort(cam.flatten(), descending=True)
    n = order.numel()
    scores = []
    for s in range(steps + 1):
        k = int(round(s / steps * n))
        mask = torch.ones(n, device=img.device)
        mask[order[:k]] = 0.0
        mask = mask.view(1, 1, H, W)
        pert = img * mask + baseline * (1 - mask)
        scores.append(F.softmax(model(pert)[0], dim=1)[0, label].item())
    return float(_trapz(scores, dx=1.0 / steps))


@torch.no_grad()
def insertion_auc(model, img, cam, label, baseline, steps=50):
    """Insert most-salient pixels first into the baseline; AUC. Higher is better."""
    H, W = cam.shape[-2:]
    order = torch.argsort(cam.flatten(), descending=True)
    n = order.numel()
    scores = []
    for s in range(steps + 1):
        k = int(round(s / steps * n))
        mask = torch.zeros(n, device=img.device)
        mask[order[:k]] = 1.0
        mask = mask.view(1, 1, H, W)
        pert = img * mask + baseline * (1 - mask)
        scores.append(F.softmax(model(pert)[0], dim=1)[0, label].item())
    return float(_trapz(scores, dx=1.0 / steps))


# --------------------------------------------------------------------------- #
#  Comprehensiveness / Sufficiency (ERASER, DeYoung et al. 2020)              #
# --------------------------------------------------------------------------- #
def _rationale_mask(cam, keep_fraction):
    """Binary mask selecting the top `keep_fraction` most-salient pixels."""
    thr = torch.quantile(cam.flatten(), 1.0 - keep_fraction)
    return (cam >= thr).float()


@torch.no_grad()
def comprehensiveness(model, img, cam, label, baseline, keep_fractions=(0.5,)):
    """Drop in probability when the rationale is *removed*. Higher is better."""
    H, W = cam.shape[-2:]
    p_full = F.softmax(model(img)[0], dim=1)[0, label].item()
    drops = []
    for kf in keep_fractions:
        m = _rationale_mask(cam, kf).view(1, 1, H, W)
        removed = img * (1 - m) + baseline * m
        p = F.softmax(model(removed)[0], dim=1)[0, label].item()
        drops.append(p_full - p)
    return float(np.mean(drops))


@torch.no_grad()
def sufficiency(model, img, cam, label, baseline, keep_fractions=(0.5,)):
    """Drop in probability when *only* the rationale is kept. Lower is better."""
    H, W = cam.shape[-2:]
    p_full = F.softmax(model(img)[0], dim=1)[0, label].item()
    drops = []
    for kf in keep_fractions:
        m = _rationale_mask(cam, kf).view(1, 1, H, W)
        kept = img * m + baseline * (1 - m)
        p = F.softmax(model(kept)[0], dim=1)[0, label].item()
        drops.append(p_full - p)
    return float(np.mean(drops))


# --------------------------------------------------------------------------- #
#  Dataset-level driver                                                        #
# --------------------------------------------------------------------------- #
def compute_metrics_on_loader(model, gradcam, dataloader, device="cuda",
                              steps=50, keep_fractions=(0.5,), baseline="black",
                              csv_path=None, progress=True):
    """
    Runs the four faithfulness metrics over a DataLoader and returns the means
    plus a per-image record list. Optionally writes a CSV.
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(dataloader, disable=not progress)
    except ImportError:
        iterator = dataloader

    model.eval()
    records = []
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        cams = compute_cams(model, gradcam, images, labels, device)  # (B,1,H,W)

        for i in range(images.shape[0]):
            img = images[i:i + 1]
            cam = cams[i, 0]                      # (H, W)
            label = int(labels[i].item())
            base = _make_baseline(img, baseline)

            records.append({
                "label": label,
                "deletion": deletion_auc(model, img, cam, label, base, steps),
                "insertion": insertion_auc(model, img, cam, label, base, steps),
                "comprehensiveness": comprehensiveness(model, img, cam, label, base, keep_fractions),
                "sufficiency": sufficiency(model, img, cam, label, base, keep_fractions),
            })

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
