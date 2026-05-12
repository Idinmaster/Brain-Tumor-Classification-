"""
Quick Inference Test - Save Visualization to File
=================================================
Test the inference visualization by saving to a file instead of displaying.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.visualize_predictions import load_model_and_config, collect_predictions
import matplotlib.pyplot as plt
import random


def save_examples_to_file(correct_examples, incorrect_examples, class_names,
                         num_display=3, output_file="inference_examples.png"):
    """Save examples to a file instead of displaying."""

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
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {output_file}")


def main():
    # Test with transformer model
    model_dir = "checkpoints/transformer"

    print("=== Testing Inference Visualization ===")

    try:
        # Load model and config
        model, config, device = load_model_and_config(model_dir)
        print(f"Loaded {config.model} model: {config.arch}")

        # Load test data
        from src.modern.dataset import get_dataloaders
        _, test_loader = get_dataloaders(
            train_dir=config.test_dir,
            test_dir=config.test_dir,
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Collect predictions
        print("Collecting prediction examples...")
        correct_examples, incorrect_examples, class_names = collect_predictions(
            model, test_loader, device, num_examples=20
        )

        print(f"Found {len(correct_examples)} correct and {len(incorrect_examples)} incorrect examples")

        # Save visualization
        save_examples_to_file(correct_examples, incorrect_examples, class_names,
                            num_display=3, output_file="inference_test.png")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())