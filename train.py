import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Trainer import Trainer  # ta classe Trainer
from differentiable_gradcam import DifferentiableGradCAM
from MaskGenerator import MaskGenerator
from Model import FullModel  # Remplace par le chemin réel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


def get_dataloader(dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == "mnist":
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor()])
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == "imagenette":
        train_dataset = datasets.ImageFolder(root="./imagenette/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./imagenette/val", transform=transform)
        num_classes = len(train_dataset.classes)
    elif dataset_name.lower() == "chestxray":
        train_dataset = datasets.ImageFolder(root="./chestxray/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./chestxray/val", transform=transform)
        num_classes = len(train_dataset.classes)
    elif dataset_name.lower() == "miniddsm":
        # Structure attendue :
        # data/miniddsm/train/Left_CC/*.png, Left_MLO/*.png, Right_CC/*.png, Right_MLO/*.png
        train_dataset = datasets.ImageFolder(root="./miniddsm/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./miniddsm/val", transform=transform)
        num_classes = len(train_dataset.classes)  # typiquement 3 : Benign, Cancer, Normal
    else:
        raise ValueError(f"Dataset {dataset_name} non reconnu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, num_classes

def get_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name in ["resnet18", "resnet50", "vgg16", "densenet121", "efficientnet_b0"]:
        model = FullModel(num_classes=num_classes, backbone_name=model_name, pretrained=True)
    else:
        raise ValueError(f"Modèle {model_name} non reconnu")
    return model.cuda()


def get_optimizer(optimizer_name, model_params, lr):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimiseur {optimizer_name} non reconnu")

def get_criterion(criterion_name):
    if criterion_name.lower() == "crossentropy":
        return nn.CrossEntropyLoss()
    elif criterion_name.lower() == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Critère {criterion_name} non reconnu")


def compute_metrics(true_labels, preds, probs, average='macro'):
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average=average, zero_division=0)
    rec = recall_score(true_labels, preds, average=average, zero_division=0)
    f1 = f1_score(true_labels, preds, average=average, zero_division=0)

    try:
        if probs.shape[1] == 2:  # binaire
            auc = roc_auc_score(true_labels, probs[:,1])
        else:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo', average=average)
    except Exception as e:
        print(f"⚠️ Impossible de calculer AUC: {e}")
        auc = None

    return acc, prec, rec, f1, auc

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Script flexible pour entraîner un modèle avec ou sans GradCAM")
    
    #Paramètres
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--criterion", type=str, default="crossentropy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mask_type", type=str, default="center", choices=["center","circle","border","diffuse","latent"])
    
    # parser.add_argument("--use_cam_loss", type=bool, default=True)
    parser.add_argument("--use_cam_loss", type=str2bool, nargs='?', const=True, default=False,
                    help="Activer ou désactiver la CAM loss (True/False)")
    parser.add_argument("--use_adaptive_supervision",type=str2bool,nargs='?',const=True,
        default=False,help="Activer la supervision Grad-CAM adaptative (alpha_t variable)"
    )

    parser.add_argument("--gradcam_loss_weight", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="./logs")

    args = parser.parse_args()

    # 🔒 Sécurité logique : supervision adaptative ⇒ CAM active
    if args.use_adaptive_supervision and not args.use_cam_loss:
        raise ValueError(
            "use_adaptive_supervision=True nécessite use_cam_loss=True"
        )

    #Affichage du mode sélectionné
    if args.use_cam_loss and args.use_adaptive_supervision:
        mode = f"⚙️ Grad-CAM supervision ADAPTATIVE avec {args.dataset} {args.mask_type}"
    elif args.use_cam_loss:
        mode = f"⚙️ Grad-CAM supervision FIXE avec {args.dataset} {args.mask_type}"
    else:
        mode = f"⚙️ Entraînement baseline (sans Grad-CAM) avec {args.dataset} {args.mask_type}"

    # mode = f"⚙️ Entraînement supervisé par Grad-CAM avec {args.dataset} {args.mask_type}" if args.use_cam_loss else "⚙️ Entraînement baseline (sans Grad-CAM)"
    print(f"\n===== MODE ACTIF : {mode} =====\n")

    #Dataset
    train_loader, val_loader, num_classes = get_dataloader(args.dataset, args.batch_size)

    #Extraire un batch fixe (utile pour GradCAM ou visualisation)
    torch.manual_seed(42)
    fixed_images, fixed_labels = next(iter(train_loader))
    fixed_images, fixed_labels = fixed_images.cuda(), fixed_labels.cuda()

    save_dir = "fixed_data"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(fixed_images.cpu(), os.path.join(save_dir, "fixed_images.pt"))
    torch.save(fixed_labels.cpu(), os.path.join(save_dir, "fixed_labels.pt"))

    print(f"✅ Premier batch sauvegardé dans {save_dir}")



    #Ajouter l'attribut name pour le Trainer
    train_loader.dataset.name = args.dataset.lower()
    val_loader.dataset.name = args.dataset.lower()

    #Modèle
    model = get_model(args.model, num_classes)

    #GradCAM Module
    gradcam_module = DifferentiableGradCAM().cuda()

    #Optimiseur et perte
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = get_criterion(args.criterion)

    #Mask Generator
    mask_gen = MaskGenerator(device='cuda')

    #Trainer
    trainer = Trainer(
        model=model,
        gradcam_module=gradcam_module,
        optimizer=optimizer,
        dataloader=train_loader,
        criterion=criterion,
        gradcam_loss_weight=args.gradcam_loss_weight,
        use_cam_loss=args.use_cam_loss,
        mask_type=args.mask_type,
        mask_generator=mask_gen,
        fixed_images=fixed_images,
        fixed_labels=fixed_labels,
        use_adaptive_supervision=args.use_adaptive_supervision,
        total_epochs=args.epochs,
 
    )

#--------------------------------------Entraînement ----------------------------------------------------------
    import time

    import json
    import numpy as np

    #Metriques lors de l'entraînement-------------------------------------------------
    train_metrics_history = []
    val_metrics_history = []
    time_history = []
    best_model_info = []

    best_f1 = -np.inf
    best_epoch = -1

    #----------------------------------------------------------------------------------

    for epoch in range(args.epochs):

        #timer epoch----------------------------
        torch.cuda.synchronize()# 🔁 attend que le GPU ait fini avant de commencer
        epoch_start = time.time()
        #----------------------------------------

        #-------TRAIN-----------------------------------------
        loss = trainer.train_epoch()
            
            # Timer epoch----------------------------
        torch.cuda.synchronize()
        epoch_end = time.time()
        time_history.append(epoch_end - epoch_start)
        print(f"⏱️ Temps pour l'époque {epoch+1}: {epoch_end - epoch_start:.2f} secondes")
            #----------------------------------------

        preds, labels, probs = trainer.evaluate_predictions()
        acc = trainer.evaluate_accuracy()
        acc_t, prec_train, rec_train, f1_train, auc_train = compute_metrics(labels, preds, probs)

    
    
                    #------------ #Affichage TRAIN ------------------------
        print(f"Epoch {epoch+1}")
        print(f""" TRAIN :  - Loss: {loss:.4f}, Acc: {acc:.4f} ou Acc : {acc_t:.4f},
        Prec: {prec_train:.4f}, Rec: {rec_train:.4f}, F1: {f1_train:.4f}, AUC: {auc_train}""")
        #------------------------------------------------------

        # ---------- VALIDATION -------------------------------------------------
        # val_loss = trainer.evaluate_epoch(val_loader)
        val_preds, val_labels, val_probs = trainer.evaluate_predictions_val(val_loader)
        acc_v, prec_val, rec_val, f1_val, auc_val = compute_metrics(val_labels, val_preds, val_probs)
        acc_val, val_loss = trainer.evaluate_accuracy_val_data(val_loader)

                    #------------ #Affichage VALIDATION ------------------------
        print(f""" VAL :  - Loss: {val_loss:.4f}, Acc: {acc_v:.4f} ou Acc : {acc_val:.4f},
        Prec: {prec_val:.4f}, Rec: {rec_val:.4f}, F1: {f1_val:.4f}, AUC: {auc_val}""")
        print("-" * 50)
        #----------------------------------------------------------------------
        # Sauvegarder le modèle si c'est le meilleur F1
        if f1_val > best_f1:
            best_f1 = f1_val
            best_epoch = epoch + 1
            # torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")
            trainer.save_full_and_state_model(trainer.model,"best_model")

            print(f"💾 Modèle sauvegardé à l'époque {best_epoch} avec F1: {best_f1:.4f}")

             # Sauvegarder les métriques
        train_metrics_history.append({
            "epoch": epoch + 1,
            "loss": float(loss),
            "accuracy": float(acc_t),
            "precision": float(prec_train),
            "recall": float(rec_train),
            "f1_score": float(f1_train),
            "auc":  float(auc_train) if auc_train is not None else None,
            "epoch_time": float(epoch_end - epoch_start)
        })
        val_metrics_history.append({
            "epoch": epoch + 1,
            "loss": float(val_loss),
            "accuracy": float(acc_v),
            "precision": float(prec_val),
            "recall": float(rec_val),
            "f1_score": float(f1_val),
            "auc": float(auc_val) if auc_val is not None else None,
            "best_f1_so_far": float(best_f1),
            "best_epoch_so_far": int(best_epoch)
        })
        #------------------------------------------------------------
    
    # ---------- SAUVEGARDE FINALE DES MÉTRIQUES ----------

    trainer.save_list_as_json(train_metrics_history, "train_metrics.json")
    trainer.save_list_as_json(val_metrics_history, "val_metrics.json")
    trainer.save_list_as_json(time_history, "training_time.json")
    best_model_info = {
            "best_f1_score": round(best_f1, 4),
            "best_epoch": best_epoch
        }
    trainer.save_list_as_json(best_model_info, "best_model_info.json")

    

    # with open("train_metrics.json", "w") as f:
    #     json.dump(train_metrics_history, f, indent=4)

    # with open("val_metrics.json", "w") as f:
    #     json.dump(val_metrics_history, f, indent=4)

    # with open("training_time.json", "w") as f:
    #     json.dump(time_history, f, indent=4)

    # with open("best_model_info.json", "w") as f:
    #     json.dump({
    #         "best_f1_score": round(best_f1, 4),
    #         "best_epoch": best_epoch
    #     }, f, indent=4)
    #------------------------------------------------------


    print(f"✅ Entraînement terminé. Meilleur modèle à l’epoch {best_epoch} avec F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
