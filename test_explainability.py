"""
test_explainability.py
-----------------------
Smoke test: verifies that explainability.py runs end-to-end with YOUR
own Model and DifferentiableGradCAM, on random data — no trained
checkpoint and no dataset required.

Run from the repo root:
    python test_explainability.py

Expected: it prints four metric values and "OK". The numbers will look
random (untrained model on random images) — that is normal. The point is
only to confirm the pipeline runs without errors before you launch a real
evaluation with evaluate.py.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from Model import FullModel
from differentiable_gradcam import DifferentiableGradCAM
from explainability import compute_metrics_on_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

torch.manual_seed(0)
N, n_classes = 8, 10
images = torch.rand(N, 3, 224, 224)
labels = torch.randint(0, n_classes, (N,))
loader = DataLoader(TensorDataset(images, labels), batch_size=4)

model = FullModel(num_classes=n_classes, backbone_name="resnet18", pretrained=False).to(device).eval()
gradcam = DifferentiableGradCAM().to(device)

means, records = compute_metrics_on_loader(
    model, gradcam, loader,
    device=device,
    steps=20,                 # low for speed
    keep_fractions=(0.5,),
    baseline="black",
    csv_path="test_scores.csv",
    progress=True,
)

print("\nResults on random data (sanity ranges, not meaningful values):")
for k, v in means.items():
    print(f"  {k:18s}: {v:.4f}")

assert len(records) == N, "wrong number of per-image records"
assert all(isinstance(v, float) for v in means.values())
print(f"\nOK — pipeline runs. Per-image CSV written to test_scores.csv ({len(records)} rows).")
