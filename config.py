"""
config.py
---------
YAML-driven configuration with CLI overrides, fully backward compatible
with the original argparse interface of train.py.

Priority order (highest wins):
    1. CLI flags explicitly passed
    2. Values from --config file (if any)
    3. Built-in defaults

If --config is NOT passed, train.py behaves exactly as before:
    python train.py --dataset cifar10 --mask_type circle --epochs 20

If --config is passed, the YAML provides defaults that CLI can still
override field-by-field:
    python train.py --config configs/cifar10_circle_adaptive.yaml --lr 5e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


# --------------------------------------------------------------------------- #
#  Built-in defaults — identical to the original train.py argparse defaults.   #
#  Single source of truth: change them here, not in train.py.                  #
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    "dataset":                  "cifar10",
    "model":                    "resnet18",
    "optimizer":                "adam",
    "criterion":                "crossentropy",
    "batch_size":               32,
    "epochs":                   20,
    "lr":                       1e-3,
    "mask_type":                "center",
    "use_cam_loss":             False,
    "use_adaptive_supervision": False,
    "gradcam_loss_weight":      1.0,
    "save_dir":                 "./logs",
    "seed":                     42,
}

VALID_MASKS = [
    "center", "circle", "border", "diffuse", "latent", "ellipse", "tissue",
    "gaussian_anisotropic", "gaussian_mixture", "radial", "directional",
    "sigmoid", "ring",
]


# --------------------------------------------------------------------------- #
#  Type coercion helpers                                                       #
# --------------------------------------------------------------------------- #
def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}.")


# --------------------------------------------------------------------------- #
#  YAML loader                                                                 #
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file. Supports either flat keys or grouped sections
    (data/model/train/supervision/output). Returns a flat dict matching DEFAULTS keys."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from e

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    flat: Dict[str, Any] = {}
    # Either grouped (data/model/train/supervision/output) or flat — accept both.
    for k, v in raw.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[sub_k] = sub_v
        else:
            flat[k] = v

    # Reject unknown keys early — typos are the #1 source of "why is my config
    # ignored?" bugs.
    unknown = set(flat) - set(DEFAULTS) - {"name", "description"}
    if unknown:
        raise ValueError(
            f"Unknown keys in {path}: {sorted(unknown)}.\n"
            f"Valid keys: {sorted(DEFAULTS)}."
        )

    # Cosmetic-only fields, not config
    flat.pop("name", None)
    flat.pop("description", None)
    return flat


# --------------------------------------------------------------------------- #
#  Parser builder                                                              #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser. All non-config flags default to argparse's
    SUPPRESS sentinel so we can detect which ones were *explicitly* passed."""
    p = argparse.ArgumentParser(
        description="Train HAtt-CNN. Accepts YAML config and/or CLI flags."
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (optional).")

    # Use SUPPRESS so unset flags do NOT land in the Namespace.
    # That lets the YAML layer fill them in, and only *explicit* CLI flags override the YAML.
    S = argparse.SUPPRESS

    p.add_argument("--dataset",                  type=str,   default=S)
    p.add_argument("--model",                    type=str,   default=S)
    p.add_argument("--optimizer",                type=str,   default=S)
    p.add_argument("--criterion",                type=str,   default=S)
    p.add_argument("--batch_size",               type=int,   default=S)
    p.add_argument("--epochs",                   type=int,   default=S)
    p.add_argument("--lr",                       type=float, default=S)
    p.add_argument("--mask_type",                type=str,   default=S, choices=VALID_MASKS)
    p.add_argument("--use_cam_loss",             type=str2bool, nargs="?", const=True, default=S)
    p.add_argument("--use_adaptive_supervision", type=str2bool, nargs="?", const=True, default=S)
    p.add_argument("--gradcam_loss_weight",      type=float, default=S)
    p.add_argument("--save_dir",                 type=str,   default=S)
    p.add_argument("--seed",                     type=int,   default=S,
                   help="Random seed. Default 42; expose only if you need it (e.g. multi-seed robustness).")
    return p


# --------------------------------------------------------------------------- #
#  Main entry: get a fully resolved namespace                                  #
# --------------------------------------------------------------------------- #
def get_config(argv=None) -> argparse.Namespace:
    """Resolve final config. Returns a Namespace with ALL keys from DEFAULTS."""
    parser = build_parser()
    cli = vars(parser.parse_args(argv))

    # 1) start from built-in defaults
    cfg: Dict[str, Any] = dict(DEFAULTS)

    # 2) overlay YAML (if any)
    config_path = cli.pop("config", None)
    if config_path:
        yaml_cfg = _load_yaml(config_path)
        cfg.update(yaml_cfg)

    # 3) overlay explicit CLI flags (anything still in `cli` was explicitly passed)
    cfg.update(cli)

    # Cross-field validation, same rule as the original train.py
    if cfg["use_adaptive_supervision"] and not cfg["use_cam_loss"]:
        raise ValueError("use_adaptive_supervision=True requires use_cam_loss=True")

    if cfg["mask_type"] not in VALID_MASKS:
        raise ValueError(
            f"Invalid mask_type {cfg['mask_type']!r}. Valid: {VALID_MASKS}"
        )

    return argparse.Namespace(**cfg)


def describe_config(cfg: argparse.Namespace, config_path: str | None = None) -> str:
    """Pretty banner shown at the top of training — makes the resolved config
    visible in logs (very useful when YAML + overrides are mixed)."""
    lines = ["=" * 62, " RESOLVED CONFIGURATION"]
    if config_path:
        lines.append(f"   source YAML : {config_path}")
    lines.append("-" * 62)
    for k in DEFAULTS:
        v = getattr(cfg, k)
        marker = "" if v == DEFAULTS[k] else "  *"
        lines.append(f"  {k:30s} = {v!r}{marker}")
    lines.append("=" * 62)
    lines.append("  (* = differs from built-in default)")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check: `python config.py --dataset miniddsm --epochs 5`
    cfg = get_config()
    print(describe_config(cfg))
