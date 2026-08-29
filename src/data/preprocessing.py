import os
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from src.logger import logger

class TBImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.classes.append(class_dir.name)
            self.class_to_idx[class_dir.name] = idx

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    self.samples.append((str(img_path), idx))

        logger.info(f"Dataset loaded from {root_dir}")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    train_transform, val_transform = get_transforms(image_size)

    full_dataset = TBImageDataset(root_dir=data_dir, transform=None)

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    test_subset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    info = {
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_size": total_size
    }

    logger.info(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
    logger.info("Data loaders created successfully")

    return train_loader, val_loader, test_loader, info