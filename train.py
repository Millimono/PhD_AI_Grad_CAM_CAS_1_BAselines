"""
train.py
--------
Entry point for HAtt-CNN training.

Three usage modes:
    (A) Legacy CLI flags (backward compatible with the original train.py):
        python train.py --dataset cifar10 --mask_type circle --use_cam_loss True --epochs 20

    (B) YAML config (recommended for reproducibility):
        python train.py --config configs/cifar10_circle_adaptive.yaml

    (C) YAML config + ad-hoc overrides:
        python train.py --config configs/cifar10_circle_adaptive.yaml --lr 5e-4

The training loop itself, metrics computation, and checkpoint saving are
UNCHANGED from the original version — only the configuration layer is new.
"""
import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from config import get_config, describe_config
from Trainer import Trainer
from differentiable_gradcam import DifferentiableGradCAM
from MaskGenerator import MaskGenerator
from Model import FullModel

sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Data / model / optimizer factories — unchanged from the original train.py  #
# --------------------------------------------------------------------------- #
def get_dataloader(dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    name = dataset_name.lower()
    if name == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "imagenette":
        train_dataset = datasets.ImageFolder(root="./imagenette/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./imagenette/val", transform=transform)
        num_classes = len(train_dataset.classes)
    elif name == "chestxray":
        train_dataset = datasets.ImageFolder(root="./chestxray/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./chestxray/val", transform=transform)
        num_classes = len(train_dataset.classes)
    elif name == "miniddsm":
        train_dataset = datasets.ImageFolder(root="./miniddsm_binary/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./miniddsm_binary/val", transform=transform)
        num_classes = len(train_dataset.classes)
    else:
        raise ValueError(f"Dataset {dataset_name} non reconnu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, num_classes


def get_model(model_name, num_classes):
    name = model_name.lower()
    if name in ["resnet18", "resnet50", "vgg16", "densenet121", "efficientnet_b0"]:
        model = FullModel(num_classes=num_classes, backbone_name=name, pretrained=True)
    else:
        raise ValueError(f"Modèle {model_name} non reconnu")
    return model.cuda()


def get_optimizer(name, model_params, lr):
    n = name.lower()
    if n == "adam":
        return optim.Adam(model_params, lr=lr)
    if n == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    raise ValueError(f"Optimiseur {name} non reconnu")


def get_criterion(name):
    n = name.lower()
    if n == "crossentropy":
        return nn.CrossEntropyLoss()
    if n == "mse":
        return nn.MSELoss()
    raise ValueError(f"Critère {name} non reconnu")


def compute_metrics(true_labels, preds, probs, average="macro"):
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average=average, zero_division=0)
    rec = recall_score(true_labels, preds, average=average, zero_division=0)
    f1 = f1_score(true_labels, preds, average=average, zero_division=0)
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(true_labels, probs[:, 1])
        else:
            auc = roc_auc_score(true_labels, probs, multi_class="ovo", average=average)
    except Exception as e:
        print(f"⚠️ Impossible de calculer AUC: {e}")
        auc = None
    return acc, prec, rec, f1, auc


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    cfg = get_config()
    config_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    print(describe_config(cfg, config_path))

    # Selected-mode banner (same format as the original)
    if cfg.use_cam_loss and cfg.use_adaptive_supervision:
        mode = f"⚙️ Grad-CAM supervision ADAPTATIVE avec {cfg.dataset} {cfg.mask_type}"
    elif cfg.use_cam_loss:
        mode = f"⚙️ Grad-CAM supervision FIXE avec {cfg.dataset} {cfg.mask_type}"
    else:
        mode = f"⚙️ Entraînement baseline (sans Grad-CAM) avec {cfg.dataset} {cfg.mask_type}"
    print(f"\n===== MODE ACTIF : {mode} =====\n")

    # ----------------- Dataset -----------------
    train_loader, val_loader, num_classes = get_dataloader(cfg.dataset, cfg.batch_size)

    # ----------------- Fixed batch for visualization -----------------
    torch.manual_seed(cfg.seed)
    fixed_images, fixed_labels = next(iter(train_loader))
    fixed_images, fixed_labels = fixed_images.cuda(), fixed_labels.cuda()
    os.makedirs("fixed_data", exist_ok=True)
    torch.save(fixed_images.cpu(), "fixed_data/fixed_images.pt")
    torch.save(fixed_labels.cpu(), "fixed_data/fixed_labels.pt")
    print(f"✅ Premier batch sauvegardé dans fixed_data")

    train_loader.dataset.name = cfg.dataset.lower()
    val_loader.dataset.name = cfg.dataset.lower()

    # ----------------- Model & Trainer -----------------
    model = get_model(cfg.model, num_classes)
    gradcam_module = DifferentiableGradCAM().cuda()
    optimizer = get_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    criterion = get_criterion(cfg.criterion)
    mask_gen = MaskGenerator(device="cuda")

    trainer = Trainer(
        model=model,
        gradcam_module=gradcam_module,
        optimizer=optimizer,
        dataloader=train_loader,
        criterion=criterion,
        gradcam_loss_weight=cfg.gradcam_loss_weight,
        use_cam_loss=cfg.use_cam_loss,
        mask_type=cfg.mask_type,
        mask_generator=mask_gen,
        fixed_images=fixed_images,
        fixed_labels=fixed_labels,
        use_adaptive_supervision=cfg.use_adaptive_supervision,
        total_epochs=cfg.epochs,
    )

    # ----------------- Training loop (unchanged) -----------------
    train_metrics_history = []
    val_metrics_history = []
    time_history = []
    best_f1 = -np.inf
    best_epoch = -1

    for epoch in range(cfg.epochs):
        torch.cuda.synchronize()
        epoch_start = time.time()

        loss = trainer.train_epoch()

        torch.cuda.synchronize()
        epoch_end = time.time()
        time_history.append(epoch_end - epoch_start)
        print(f"⏱️ Temps pour l'époque {epoch + 1}: {epoch_end - epoch_start:.2f} secondes")

        preds, labels, probs = trainer.evaluate_predictions()
        acc = trainer.evaluate_accuracy()
        acc_t, prec_train, rec_train, f1_train, auc_train = compute_metrics(labels, preds, probs)

        print(f"Epoch {epoch + 1}")
        print(f" TRAIN : - Loss: {loss:.4f}, Acc: {acc:.4f} ou Acc : {acc_t:.4f},\n"
              f"  Prec: {prec_train:.4f}, Rec: {rec_train:.4f}, F1: {f1_train:.4f}, AUC: {auc_train}")

        val_preds, val_labels, val_probs = trainer.evaluate_predictions_val(val_loader)
        acc_v, prec_val, rec_val, f1_val, auc_val = compute_metrics(val_labels, val_preds, val_probs)
        acc_val, val_loss = trainer.evaluate_accuracy_val_data(val_loader)

        print(f" VAL : - Loss: {val_loss:.4f}, Acc: {acc_v:.4f} ou Acc : {acc_val:.4f},\n"
              f"  Prec: {prec_val:.4f}, Rec: {rec_val:.4f}, F1: {f1_val:.4f}, AUC: {auc_val}")
        print("-" * 50)

        if f1_val > best_f1:
            best_f1 = f1_val
            best_epoch = epoch + 1
            trainer.save_full_and_state_model(trainer.model, "best_model")
            print(f"💾 Modèle sauvegardé à l'époque {best_epoch} avec F1: {best_f1:.4f}")

        train_metrics_history.append({
            "epoch": epoch + 1, "loss": float(loss),
            "accuracy": float(acc_t), "precision": float(prec_train),
            "recall": float(rec_train), "f1_score": float(f1_train),
            "auc": float(auc_train) if auc_train is not None else None,
            "epoch_time": float(epoch_end - epoch_start),
        })
        val_metrics_history.append({
            "epoch": epoch + 1, "loss": float(val_loss),
            "accuracy": float(acc_v), "precision": float(prec_val),
            "recall": float(rec_val), "f1_score": float(f1_val),
            "auc": float(auc_val) if auc_val is not None else None,
            "best_f1_so_far": float(best_f1), "best_epoch_so_far": int(best_epoch),
        })

    trainer.save_list_as_json(train_metrics_history, "train_metrics.json")
    trainer.save_list_as_json(val_metrics_history, "val_metrics.json")
    trainer.save_list_as_json(time_history, "training_time.json")
    trainer.save_list_as_json(
        {"best_f1_score": round(best_f1, 4), "best_epoch": best_epoch},
        "best_model_info.json",
    )

    print(f"✅ Entraînement terminé. Meilleur modèle à l'epoch {best_epoch} avec F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
