from torch.utils.data import Dataset
from PIL import Image

class MiniDDSM_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # supposer que le masque a le même nom mais dans mask/
                mask_path = img_path.replace("train", "masks")
                self.samples.append((img_path, class_name, mask_path))

        self.class_to_idx = {cls: i for i, cls in enumerate(sorted({s[1] for s in self.samples}))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # masque en grayscale
        label = self.class_to_idx[class_name]

        if self.transform:
            img = self.transform(img)
            mask = transforms.ToTensor()(mask)  # convert mask en tensor

        return img, label, mask



""""
elif dataset_name.lower() == "miniddsm":
    train_dataset = MiniDDSM_Dataset("./data/miniddsm/train", transform=transform)
    val_dataset = MiniDDSM_Dataset("./data/miniddsm/val", transform=transform)
    num_classes = len(train_dataset.class_to_idx)

    data/
├── miniddsm/
│   ├── train/
│   │   ├── Benign/
│   │   ├── Cancer/
│   │   └── Normal/
│   ├── val/
│   │   ├── Benign/
│   │   ├── Cancer/
│   │   └── Normal/
│   └── masks/
│       ├── train/
│       └── val/

"""