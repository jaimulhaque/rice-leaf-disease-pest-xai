import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


DATA_DIR = Path("/mnt/g/rice-leaf-disease-xai/data/Rice Dataset/Original Dataset")

CLASSES = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "Rice"])
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

VALID_EXT = [".jpg", ".jpeg", ".png"]

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gather_samples():
    """Return list of (filepath, label_idx) for all 7 classes."""
    samples = []
    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        for f in cls_dir.iterdir():
            if f.suffix.lower() in VALID_EXT:
                samples.append((str(f), CLASS_TO_IDX[cls]))
    return samples


def stratified_split(samples, seed=42):
    """70/15/15 stratified split."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    train_set = list(zip(train_paths, train_labels))
    val_set = list(zip(val_paths, val_labels))
    test_set = list(zip(test_paths, test_labels))
    return train_set, val_set, test_set


def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class RiceLeafDataset(Dataset):
    def __init__(self, samples, split="train"):
        self.samples = samples
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  # force RGB (handles RGBA PNGs)
        img = self.transform(img)
        return img, label


def compute_class_weights(train_set):
    """Inverse-frequency class weights for weighted loss (handles imbalance)."""
    labels = [lbl for _, lbl in train_set]
    counts = np.bincount(labels, minlength=len(CLASSES))
    total = counts.sum()
    weights = total / (len(CLASSES) * counts)
    return weights.astype(np.float32)


if __name__ == "__main__":
    samples = gather_samples()
    print("Total samples (7 classes):", len(samples))

    train_set, val_set, test_set = stratified_split(samples)
    print("Train:", len(train_set), "Val:", len(val_set), "Test:", len(test_set))

    weights = compute_class_weights(train_set)
    for cls, w in zip(CLASSES, weights):
        print(f"{cls}: weight={w:.3f}")
