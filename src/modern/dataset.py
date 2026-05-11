"""
Shared Dataset & DataLoader Utilities
======================================
Used by both CNN and Transformer training scripts.
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Augmented transforms for the training set."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for val/test sets."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    train_dir: str,
    test_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and testing DataLoaders from ImageFolder-structured directories.

    Expected layout:
        <train_dir>/
            glioma/
            meningioma/
            notumor/
            pituitary/
        <test_dir>/   (same structure)

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(img_size),
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_val_transforms(img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Dataset]  Train: {len(train_dataset)} images | "
          f"Test: {len(test_dataset)} images")
    print(f"[Dataset]  Classes: {train_dataset.classes}")

    return train_loader, test_loader


def get_train_val_loaders(
    train_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 1 / 6,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split the training directory into a training and validation set.

    Uses a 5:1 train/validation split by default.
    """
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(img_size),
    )
    val_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_val_transforms(img_size),
    )

    total = len(train_dataset)
    if total == 0:
        raise ValueError(f"No images found in training directory: {train_dir}")

    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size
    if train_size == 0:
        raise ValueError("Train split is empty. Reduce val_fraction.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Dataset]  Train: {len(train_loader.dataset)} images | "
          f"Val: {len(val_loader.dataset)} images")
    print(f"[Dataset]  Classes: {train_dataset.classes}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test — adjust paths as needed
    ROOT = Path(__file__).resolve().parents[3]
    train_loader, test_loader = get_dataloaders(
        train_dir=str(ROOT / "dataset/processed/training"),
        test_dir=str(ROOT / "dataset/processed/testing"),
    )
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}  Labels: {labels}")
