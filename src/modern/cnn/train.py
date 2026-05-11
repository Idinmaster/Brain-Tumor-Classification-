"""
CNN Training Script — Brain Tumor Classification
==================================================
Usage:
    python -m src.modern.cnn.train [options]

    or from project root:
    python src/modern/cnn/train.py --arch resnet18 --epochs 30

Architectures: simple_cnn | resnet18 | resnet50 | efficientnet_b0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Allow running as a script from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modern.cnn.model import get_cnn_model
from src.modern.dataset import get_dataloaders, CLASSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float, list, list]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[CNN] Device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = get_cnn_model(
        architecture=args.arch,
        num_classes=len(CLASSES),
        pretrained=args.pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CNN] Architecture: {args.arch}  |  Trainable params: {total_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'LR':>10} {'Time':>8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>10.4f} {current_lr:>10.2e} {elapsed:>7.1f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = save_dir / f"cnn_{args.arch}_best.pth"
            torch.save({
                "epoch": epoch,
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  ✓ Best model saved → {ckpt_path}  (val_acc={val_acc:.4f})")

    # Final evaluation with classification report
    print(f"\n[CNN] Best Validation Accuracy: {best_acc:.4f}")
    print("\n[CNN] Final Evaluation on Test Set:")
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # Save history
    np.save(save_dir / f"cnn_{args.arch}_history.npy", history)
    print(f"[CNN] Training history saved → {save_dir}/cnn_{args.arch}_history.npy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Train CNN for brain tumor classification")
    parser.add_argument("--arch",         type=str,   default="resnet18",
                        choices=["simple_cnn", "resnet18", "resnet50", "efficientnet_b0"])
    parser.add_argument("--pretrained",   action="store_true", default=True)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--img_size",     type=int,   default=224)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--train_dir",    type=str,
                        default=str(root / "dataset/processed/training"))
    parser.add_argument("--test_dir",     type=str,
                        default=str(root / "dataset/processed/testing"))
    parser.add_argument("--save_dir",     type=str,
                        default=str(root / "checkpoints/cnn"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
