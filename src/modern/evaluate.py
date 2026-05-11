"""Evaluate a saved CNN or Transformer checkpoint using cross entropy loss."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modern.cnn.model import get_cnn_model
from src.modern.transformer.model import get_transformer_model
from src.modern.dataset import get_dataloaders, CLASSES
from src.modern.utils import ModelConfig


def get_device() -> torch.device:
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("[Device] TPU detected and selected for evaluation.")
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


def build_model(config: ModelConfig) -> nn.Module:
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


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def evaluate_checkpoint(config: ModelConfig, checkpoint_path: str) -> None:
    device = get_device()
    model = build_model(config).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    _, test_loader = get_dataloaders(
        train_dir=config.test_dir,
        test_dir=config.test_dir,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Architecture    : {config.model} / {config.arch}")
    print(f"Test samples    : {total}")
    print(f"Cross entropy   : {avg_loss:.4f}")
    print(f"Accuracy        : {accuracy:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved CNN or Transformer checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--config", type=str,
                        help="Optional ModelConfig JSON file saved during training")
    parser.add_argument("--model", type=str, choices=["cnn", "transformer"], default="cnn",
                        help="Model type if config is not provided")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Architecture name if config is not provided")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Load pretrained backbone when rebuilding the model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "dataset/processed/training")))
    parser.add_argument("--test_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "dataset/processed/testing")))
    parser.add_argument("--save_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "checkpoints")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        config = ModelConfig(
            model=args.model,
            arch=args.arch,
            pretrained=args.pretrained,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            save_dir=args.save_dir,
        )
    evaluate_checkpoint(config, args.checkpoint)
