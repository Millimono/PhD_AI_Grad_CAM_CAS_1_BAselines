"""
evaluate.py
-----------
Load a trained model and compute the four faithfulness metrics on its
validation set, reusing your existing data pipeline (train.get_dataloader)
and Grad-CAM module.

Example
-------
    python evaluate.py \
        --checkpoint logs/supervision_fixe/models/cifar10/cam_supervised/circle/best_model.pt \
        --dataset cifar10 --batch_size 32 --steps 50 --threshold 0.5 \
        --baseline black --out logs/explainability
"""
import argparse
import json
import os

import torch

from differentiable_gradcam import DifferentiableGradCAM
from explainability import compute_metrics_on_loader

# Reuse the EXACT dataset logic used during training (no duplication).
from train import get_dataloader


def load_full_model(path, device):
    """best_model.pt is saved as a full model object (torch.save(model))."""
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model


def main():
    p = argparse.ArgumentParser(description="Faithfulness evaluation of Grad-CAM maps")
    p.add_argument("--checkpoint", required=True, help="path to best_model.pt (full model)")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=50, help="deletion/insertion granularity")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="rationale fraction for comprehensiveness/sufficiency")
    p.add_argument("--eraser_bins", action="store_true",
                   help="aggregate comp/suff over the standard {0.01,0.05,0.1,0.2,0.5} bins")
    p.add_argument("--baseline", choices=["black", "blur"], default="black")
    p.add_argument("--out", default="logs/explainability")
    p.add_argument("--max_batches", type=int, default=None,
                   help="evaluate only the first N batches (quick check)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = get_dataloader(args.dataset, args.batch_size)
    val_loader.dataset.name = args.dataset.lower()

    if args.max_batches is not None:
        from itertools import islice

        class _Limited:
            def __init__(self, loader, n): self.loader, self.n = loader, n
            def __iter__(self): return islice(iter(self.loader), self.n)
        val_loader = _Limited(val_loader, args.max_batches)

    model = load_full_model(args.checkpoint, device)
    gradcam = DifferentiableGradCAM().to(device)

    keep = (0.01, 0.05, 0.1, 0.2, 0.5) if args.eraser_bins else (args.threshold,)

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.dataset}_{args.baseline}"
    means, records = compute_metrics_on_loader(
        model, gradcam, val_loader, device=device,
        steps=args.steps, keep_fractions=keep, baseline=args.baseline,
        csv_path=os.path.join(args.out, f"per_image_{tag}.csv"),
    )

    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump({"config": vars(args), "n_images": len(records), "means": means}, f, indent=4)

    print("\n===== Faithfulness metrics =====")
    print(f"Deletion AUC        : {means['deletion']:.4f}   (lower is better)")
    print(f"Insertion AUC       : {means['insertion']:.4f}   (higher is better)")
    print(f"Comprehensiveness   : {means['comprehensiveness']:.4f}   (higher is better)")
    print(f"Sufficiency         : {means['sufficiency']:.4f}   (lower is better)")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
