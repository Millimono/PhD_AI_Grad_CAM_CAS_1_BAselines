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
    "--save_dir", "./logs"
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
