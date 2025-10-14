import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision.datasets.folder import ImageFolder

def load_dataset(name, train=True, data_dir="data", image_size=224, batch_size=32):
    """
    Charge dynamiquement un dataset parmi : cifar10, isic, mvtec, fer2013
    """
    os.makedirs(data_dir, exist_ok=True)
    name = name.lower()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    elif name == "fer2013":
        dataset = datasets.FER2013(root=data_dir, train=train, download=True, transform=transform)

    elif name == "mvtec":
        #  MVTec-AD est décompressé dans data/mvtec/
        dataset_path = os.path.join(data_dir, "mvtec", "train" if train else "test")
        dataset = ImageFolder(root=dataset_path, transform=transform)

    elif name == "isic":
        dataset_path = os.path.join(data_dir, "isic", "train" if train else "test")
        dataset = ImageFolder(root=dataset_path, transform=transform)

    else:
        raise ValueError(f"Dataset '{name}' non reconnu.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    print(f"✅ Dataset chargé : {name} ({len(dataset)} images)")
    return dataloader



# Example usage:
if __name__ == "__main__":
    train_loader = load_dataset("cifar10", train=True, data_dir="./data", image_size=224, batch_size=32)
    test_loader = load_dataset("cifar10", train=False, data_dir="./data", image_size=224, batch_size=32)

    # Exemple 1 : CIFAR-10
    train_loader = load_dataset("cifar10", train=True)
    test_loader = load_dataset("cifar10", train=False)

    # Exemple 2 : ISIC
    # ( dossier data/isic/train/... et data/isic/test/...)
    isic_loader = load_dataset("isic", train=True)

    # Exemple 3 : MVTec-AD
    mvtec_loader = load_dataset("mvtec", train=False)

    # Exemple 4 : FER2013
    fer_loader = load_dataset("fer2013", train=True)
