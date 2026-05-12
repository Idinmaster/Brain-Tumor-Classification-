"""
Inference Script - Visualize Model Predictions
===============================================
Load a trained model and show examples of correct and incorrect predictions.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modern.cnn.model import get_cnn_model
from src.modern.transformer.model import get_transformer_model
from src.modern.dataset import get_dataloaders, CLASSES, get_val_transforms
from src.modern.utils import ModelConfig


def get_device() -> torch.device:
    """Get the best available device."""
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


def load_model_and_config(checkpoint_dir: str) -> Tuple[nn.Module, ModelConfig, torch.device]:
    """Load model and config from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Load config
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ModelConfig.load(str(config_path))

    # Build model
    if config.model == "cnn":
        model = get_cnn_model(
            architecture=config.arch,
            num_classes=len(CLASSES),
            pretrained=config.pretrained,
        )
    else:  # transformer
        model = get_transformer_model(
            architecture=config.arch,
            num_classes=len(CLASSES),
            pretrained=config.pretrained,
        )

    # Load checkpoint
    device = get_device()
    model = model.to(device)

    checkpoint_path = None
    # Look for best model checkpoint
    for pattern in ["*best*.pth", "*.pth"]:
        candidates = list(checkpoint_dir.glob(pattern))
        if candidates:
            checkpoint_path = candidates[0]  # Take first match
            break

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, config, device


def collect_predictions(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                       device: torch.device, num_examples: int = 50) -> Tuple[List, List, List]:
    """
    Collect predictions and return examples of correct/incorrect classifications.

    Returns:
        correct_examples: List of (image, true_label, pred_label) tuples
        incorrect_examples: List of (image, true_label, pred_label) tuples
        class_names: List of class names
    """
    correct_examples = []
    incorrect_examples = []

    # Get transforms for denormalization
    val_transforms = get_val_transforms(img_size=224)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Convert to CPU for processing
            images_cpu = images.cpu()
            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()

            for i in range(len(images_cpu)):
                img = images_cpu[i]
                true_label = labels_cpu[i].item()
                pred_label = preds_cpu[i].item()

                # Denormalize image for display
                img_denorm = img.clone()
                for c in range(3):  # RGB channels
                    img_denorm[c] = img_denorm[c] * torch.tensor([0.229, 0.224, 0.225])[c] + torch.tensor([0.485, 0.456, 0.406])[c]
                img_denorm = torch.clamp(img_denorm, 0, 1)

                # Convert to numpy and transpose to HWC
                img_np = img_denorm.numpy().transpose(1, 2, 0)

                if true_label == pred_label:
                    if len(correct_examples) < num_examples:
                        correct_examples.append((img_np, true_label, pred_label))
                else:
                    if len(incorrect_examples) < num_examples:
                        incorrect_examples.append((img_np, true_label, pred_label))

            # Stop if we have enough examples
            if len(correct_examples) >= num_examples and len(incorrect_examples) >= num_examples:
                break

    return correct_examples, incorrect_examples, CLASSES


def display_examples(correct_examples: List, incorrect_examples: List,
                    class_names: List, num_display: int = 3):
    """Display examples of correct and incorrect predictions."""

    # Select random examples
    correct_sample = random.sample(correct_examples, min(num_display, len(correct_examples)))
    incorrect_sample = random.sample(incorrect_examples, min(num_display, len(incorrect_examples)))

    fig, axes = plt.subplots(2, num_display, figsize=(15, 8))

    # Display correct predictions
    for i, (img, true_label, pred_label) in enumerate(correct_sample):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Correct\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                           color='green', fontsize=10)
        axes[0, i].axis('off')

    # Display incorrect predictions
    for i, (img, true_label, pred_label) in enumerate(incorrect_sample):
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Incorrect\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                           color='red', fontsize=10)
        axes[1, i].axis('off')

    # Set row titles
    axes[0, 0].set_ylabel('Correct Predictions', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Incorrect Predictions', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model checkpoint directory (e.g., checkpoints/transformer)")
    parser.add_argument("--num_display", type=int, default=3,
                        help="Number of examples to display for each category")
    parser.add_argument("--num_examples", type=int, default=50,
                        help="Number of examples to collect for sampling")

    args = parser.parse_args()

    print("=== Model Inference Visualization ===")
    print(f"Model directory: {args.model_dir}")
    print(f"Examples to collect: {args.num_examples}")
    print(f"Examples to display: {args.num_display}")
    print()

    try:
        # Load model and config
        model, config, device = load_model_and_config(args.model_dir)

        print(f"Loaded {config.model} model: {config.arch}")
        print(f"Using device: {device}")
        print()

        # Load test data
        _, test_loader = get_dataloaders(
            train_dir=config.test_dir,  # Using test_dir for both since we only need test
            test_dir=config.test_dir,
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Collect predictions
        print("Collecting prediction examples...")
        correct_examples, incorrect_examples, class_names = collect_predictions(
            model, test_loader, device, args.num_examples
        )

        print(f"Found {len(correct_examples)} correct and {len(incorrect_examples)} incorrect examples")
        print()

        # Display examples
        if len(correct_examples) >= args.num_display and len(incorrect_examples) >= args.num_display:
            print("Displaying prediction examples...")
            display_examples(correct_examples, incorrect_examples, class_names, args.num_display)
        else:
            print(f"Warning: Not enough examples found. Need at least {args.num_display} of each type.")
            print(f"Available: {len(correct_examples)} correct, {len(incorrect_examples)} incorrect")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())