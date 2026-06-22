"""
evaluate.py
-----------
Compute the four faithfulness metrics (deletion / insertion /
comprehensiveness / sufficiency) on a trained checkpoint.

Three usage modes:
    (A) Legacy CLI flags — exactly as before:
        python evaluate.py --checkpoint best_model.pt --dataset cifar10 --steps 50

    (B) YAML config:
        python evaluate.py --config configs/eval_cifar10.yaml

    (C) YAML + CLI override:
        python evaluate.py --config configs/eval_cifar10.yaml --max_batches 5

Evaluation YAMLs accept the following keys (flat or grouped):
    checkpoint, dataset, batch_size, steps, threshold, eraser_bins,
    baseline, out, max_batches
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from differentiable_gradcam import DifferentiableGradCAM
from explainability import compute_metrics_on_loader
from train import get_dataloader


EVAL_DEFAULTS: Dict[str, Any] = {
    "checkpoint":  None,           # required
    "dataset":     "cifar10",
    "batch_size":  32,
    "steps":       50,
    "threshold":   0.5,
    "eraser_bins": False,
    "baseline":    "black",        # "black" | "blur"
    "out":         "logs/explainability",
    "max_batches": None,
}


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}.")


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required for --config. pip install pyyaml") from e
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    flat: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    flat.pop("name", None); flat.pop("description", None)
    unknown = set(flat) - set(EVAL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}.")
    return flat


def get_eval_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Faithfulness evaluation of Grad-CAM maps.")
    p.add_argument("--config", type=str, default=None)
    S = argparse.SUPPRESS
    p.add_argument("--checkpoint",  type=str,   default=S)
    p.add_argument("--dataset",     type=str,   default=S)
    p.add_argument("--batch_size",  type=int,   default=S)
    p.add_argument("--steps",       type=int,   default=S)
    p.add_argument("--threshold",   type=float, default=S,
                   help="rationale fraction for comprehensiveness/sufficiency")
    p.add_argument("--eraser_bins", type=_str2bool, nargs="?", const=True, default=S,
                   help="aggregate comp/suff over {0.01,0.05,0.1,0.2,0.5}")
    p.add_argument("--baseline",    choices=["black", "blur"], default=S)
    p.add_argument("--out",         type=str,   default=S)
    p.add_argument("--max_batches", type=int,   default=S,
                   help="evaluate only the first N batches (quick check)")
    cli = vars(p.parse_args())

    cfg = dict(EVAL_DEFAULTS)
    config_path = cli.pop("config", None)
    if config_path:
        cfg.update(_load_yaml(config_path))
    cfg.update(cli)

    if cfg["checkpoint"] is None:
        p.error("--checkpoint is required (either via CLI or YAML)")
    return argparse.Namespace(**cfg)


def load_full_model(path, device):
    """best_model.pt is saved as a full model object (torch.save(model))."""
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model


def main():
    args = get_eval_config()
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

    summary = {"config": vars(args), "n_images": len(records), "means": means}
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n===== Faithfulness metrics =====")
    print(f"Deletion AUC        : {means['deletion']:.4f}   (lower is better)")
    print(f"Insertion AUC       : {means['insertion']:.4f}   (higher is better)")
    print(f"Comprehensiveness   : {means['comprehensiveness']:.4f}   (higher is better)")
    print(f"Sufficiency         : {means['sufficiency']:.4f}   (lower is better)")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
