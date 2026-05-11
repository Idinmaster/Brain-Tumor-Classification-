"""
Transformer Training Script — Brain Tumor Classification
=========================================================
Usage:
    python src/modern/transformer/train.py --arch swin_t --epochs 30

Architectures: mini_vit | vit_b16 | swin_t
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modern.transformer.model import get_transformer_model
from src.modern.dataset import get_dataloaders, CLASSES

# ---------------------------------------------------------------------------
# Helpers  (identical contract to the CNN versions)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping — especially helpful for ViT
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Warm-up + Cosine scheduler helper
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up for `warmup_epochs`, then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch + 1
        if ep <= self.warmup_epochs:
            factor = ep / max(1, self.warmup_epochs)
        else:
            progress = (ep - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.min_lr / self.base_lrs[0] + 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (
                1 + torch.cos(torch.tensor(progress * 3.14159)).item()
            )
        return [base_lr * factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        "cuda"  if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Transformer] Device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = get_transformer_model(
        architecture=args.arch,
        num_classes=len(CLASSES),
        pretrained=args.pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Transformer] Architecture: {args.arch}  |  Trainable params: {total_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-6,
    )

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

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = save_dir / f"transformer_{args.arch}_best.pth"
            torch.save({
                "epoch": epoch,
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  ✓ Best model saved → {ckpt_path}  (val_acc={val_acc:.4f})")

    # Final report
    print(f"\n[Transformer] Best Validation Accuracy: {best_acc:.4f}")
    print("\n[Transformer] Final Evaluation on Test Set:")
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    np.save(save_dir / f"transformer_{args.arch}_history.npy", history)
    print(f"[Transformer] History saved → {save_dir}/transformer_{args.arch}_history.npy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Train Transformer for brain tumor classification")
    parser.add_argument("--arch",         type=str,   default="swin_t",
                        choices=["mini_vit", "vit_b16", "swin_t"])
    parser.add_argument("--pretrained",   action="store_true", default=True)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--img_size",     type=int,   default=224)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--train_dir",    type=str,
                        default=str(root / "dataset/processed/training"))
    parser.add_argument("--test_dir",     type=str,
                        default=str(root / "dataset/processed/testing"))
    parser.add_argument("--save_dir",     type=str,
                        default=str(root / "checkpoints/transformer"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
