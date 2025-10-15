# Grad-CAM Supervised Training Framework

## Description

Ce projet implémente un framework flexible pour entraîner des modèles de vision par ordinateur avec supervision via Grad-CAM différentiable. Il supporte plusieurs backbones pré-entraînés (ResNet, VGG, DenseNet, EfficientNet) et permet l'utilisation de masques adaptatifs pour guider l'attention du modèle.

## Fonctionnalités

* Entraînement de modèles CNN classiques et modernes avec supervision Grad-CAM.
* Modules Grad-CAM différentiables.
* Génération de masques adaptatifs : `center`, `circle`, `border`, `diffuse`, `latent`.
* Support pour divers datasets : CIFAR-10, MNIST, ImageNette, ISIC, MVTec-AD, FER2013.
* Sauvegarde automatique des modèles et métriques.
* Visualisation Grad-CAM côte à côte avec l'image originale.

## Installation

1. Cloner le dépôt :

```bash
git clone <URL_DU_DEPOT>
cd <REPERTOIRE_DU_PROJET>
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Structure des fichiers

* `train.py` : script principal pour lancer l'entraînement.
* `run.py` : exemple d'utilisation avec arguments prédéfinis.
* `Model.py` : définition des modèles et backbones.
* `differentiable_gradcam.py` : module Grad-CAM différentiable.
* `MaskGenerator.py` : génération de masques adaptatifs.
* `Trainer.py` : classe Trainer pour gérer l'entraînement et la visualisation.

## Usage

### Entraînement de base :

```bash
python train.py --dataset cifar10 --model resnet18 --epochs 20 --batch_size 32
```

### Activer la supervision Grad-CAM :

```bash
python train.py --dataset cifar10 --model resnet18 --use_cam_loss True --gradcam_loss_weight 1.0
```

### Paramètres principaux

* `--dataset` : nom du dataset (`cifar10`, `mnist`, `imagenette`, `chestxray`).
* `--model` : nom du backbone (`resnet18`, `resnet50`, `vgg16`, `densenet121`, `efficientnet_b0`).
* `--epochs` : nombre d'époques.
* `--batch_size` : taille de batch.
* `--lr` : taux d'apprentissage.
* `--use_cam_loss` : activer/désactiver la perte Grad-CAM.
* `--gradcam_loss_weight` : poids de la perte Grad-CAM.
* `--mask_type` : type de masque (`center`, `circle`, `border`, `diffuse`).

## Visualisation

* Les Grad-CAMs sont sauvegardées dans `./logs/gradcam_analysis/{dataset}/{mode}/{mask_type}/`.
* Chaque image est sauvegardée côte à côte avec l'image originale.

## Sauvegarde

* Meilleur modèle basé sur le F1-score : `best_model.pth`.
* Historique des métriques d'entraînement : `train_metrics.json`, `val_metrics.json`, `training_time.json`, `best_model_info.json`.

## Exemple rapide

```bash
python run.py
```

Lance un entraînement sur CIFAR-10 avec ResNet18 sans supervision Grad-CAM.

## Remarques

* Le GPU est recommandé pour l'entraînement.
* Assurez-vous que les dossiers pour les datasets comme ISIC ou MVTec sont correctement placés sous `./data`.
* Les masques adaptatifs peuvent être personnalisés via `MaskGenerator.py`.
