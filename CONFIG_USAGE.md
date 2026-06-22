## Configuration: YAML + CLI

HAtt-CNN supports three usage modes, in increasing order of reproducibility:

### (A) Legacy CLI — backward compatible

The original argparse interface still works as-is:

```bash
python train.py --dataset cifar10 --mask_type circle \
                --use_cam_loss True --epochs 20
```

### (B) YAML config — recommended

Define your experiment once in a versionable file, then reproduce it
indefinitely:

```bash
python train.py --config configs/cifar10_circle_adaptive.yaml
python evaluate.py --config configs/eval_cifar10_default.yaml
```

### (C) YAML + CLI overrides

The YAML provides defaults; any CLI flag overrides the matching YAML
key. Useful for quick sweeps without editing the YAML:

```bash
python train.py --config configs/cifar10_circle_adaptive.yaml --lr 5e-4
python evaluate.py --config configs/eval_cifar10_default.yaml --max_batches 5
```

### Provided templates

| File | Purpose |
|---|---|
| `configs/cifar10_baseline.yaml`         | Reference run without attention supervision |
| `configs/cifar10_circle_adaptive.yaml`  | Best explainability run (paper: comprehensiveness ≈ 0.83) |
| `configs/miniddsm_border_fixed.yaml`    | Best classification run on MiniDDSM (paper: +1.23 acc pts) |
| `configs/eval_cifar10_default.yaml`     | Default evaluation protocol |

### YAML format

Both flat and grouped layouts are accepted. Typos in keys are caught at
load time with the list of valid keys.

```yaml
# Flat
dataset: cifar10
mask_type: circle
lr: 0.001

# Grouped (equivalent, more readable)
data:
  dataset: cifar10
supervision:
  mask_type: circle
train:
  lr: 0.001
```
