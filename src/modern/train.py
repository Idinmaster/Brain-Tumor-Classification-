"""
Unified Modern Training Script
==============================
Train either a CNN or Transformer model for brain tumor classification.

Configuration is managed through `src.modern.utils.ModelConfig`.
The config is persisted to `checkpoints/<model>/config.json` before training.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modern.cnn.model import get_cnn_model
from src.modern.transformer.model import get_transformer_model
from src.modern.dataset import get_dataloaders, get_train_val_loaders, CLASSES
from src.modern.utils import ModelConfig


def get_device() -> torch.device:
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("[Device] TPU detected and selected.")
        return device
    except ImportError:
        pass
    except Exception as exc:
        print(f"[Device] TPU detection failed: {exc}. Falling back.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    clip_grad: bool = False,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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


def plot_loss_decay(history: dict, save_dir: Path, model: str, arch: str) -> Path:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", marker="o")
    plt.title(f"Loss Decay: {model.upper()} {arch}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    fig_path = save_dir / f"{model}_{arch}_loss_decay.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            factor = epoch / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.min_lr / self.base_lrs[0] + 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (
                1 + torch.cos(torch.tensor(progress * 3.14159)).item()
            )
        return [base_lr * factor for base_lr in self.base_lrs]


def build_model(config):
    if config.model == "cnn":
        return get_cnn_model(
            architecture=config.arch,
            num_classes=len(CLASSES),
            pretrained=config.pretrained,
        )
    return get_transformer_model(
        architecture=config.arch,
        num_classes=len(CLASSES),
        pretrained=config.pretrained,
    )


def build_scheduler(optimizer, config):
    if config.model == "transformer":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs,
            min_lr=config.min_lr,
        )
    return CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)


def train(config):
    device = get_device()
    print(f"[{config.model.upper()}] Device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        train_dir=config.train_dir,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_fraction=config.val_fraction,
    )
    _, test_loader = get_dataloaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{config.model.upper()}] Architecture: {config.arch}  |  Trainable params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(optimizer, config)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'LR':>10} {'Time':>8}")
    print("-" * 75)

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False),
            criterion,
            optimizer,
            device,
            clip_grad=(config.model == "transformer"),
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            tqdm(val_loader, desc=f"Valid {epoch}/{config.epochs}", leave=False),
            criterion,
            device,
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
            best_path = save_dir / f"{config.model}_{config.arch}_best.pth"
            torch.save({
                "epoch": epoch,
                "model": config.model,
                "arch": config.arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_path)
            print(f"  ✓ Best model saved → {best_path}  (val_acc={val_acc:.4f})")

    print(f"\n[{config.model.upper()}] Best Validation Accuracy: {best_acc:.4f}")
    print(f"\n[{config.model.upper()}] Final Evaluation on Test Set:")
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    history_path = save_dir / f"{config.model}_{config.arch}_history.npy"
    np.save(history_path, history)
    print(f"[{config.model.upper()}] History saved → {history_path}")

    fig_path = plot_loss_decay(history, save_dir, config.model, config.arch)
    print(f"[{config.model.upper()}] Loss decay plot saved → {fig_path}")


if __name__ == "__main__":
    config = ModelConfig()
    config_path = config.save()
    print(f"[Config] Saved configuration to {config_path}")
    train(config)
