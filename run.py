# run.py
import sys
from train import main

# ⚙️ Arguments 
sys.argv = [
    "train.py",                # nom du script
    "--dataset", "cifar10",
    "--model", "resnet18",
    "--mask_type", "center",
    "--use_cam_loss", "False",
    "--batch_size", "32",
    "--epochs", "20",
    "--optimizer", "adam",
    "--lr", "1e-4",
    "--gradcam_loss_weight", "1.0",
    "--save_dir", "./logs",
    "--use_adaptive_supervision", "False",

]

if __name__ == "__main__":
    main()

# python3 train.py \
#   --dataset cifar10 \
#   --model resnet18 \
#   --mask_type center \
#   --use_cam_loss True \
#   --batch_size 32 \
#   --epochs 20 \
#   --lr 0.001 \
#   --gradcam_loss_weight 1.0 \
#   --save_dir ./logs


# sys.argv = [
#     "train.py",
#     "--dataset",                  "miniddsm",
#     "--model",                    "resnet18",
#     "--mask_type",                "tissue",
#     "--use_cam_loss",             "True",
#     "--use_adaptive_supervision", "False",
#     "--batch_size",               "32",
#     "--epochs",                   "50",       # ← was 20/100
#     "--optimizer",                "adam",
#     "--lr",                       "1e-5",     # ← was 1e-4, 10x plus petit
#     "--gradcam_loss_weight",      "0.1",      # ← was 1.0, moins agressif
#     "--save_dir",                 "./logs",
# ]