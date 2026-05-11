"""
Naive Bayes Training Script
===========================
Train a Naive Bayes baseline model for brain tumor classification.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.naive_bayes.model import train_naive_bayes


def main():
    parser = argparse.ArgumentParser(description="Train Naive Bayes baseline model")
    parser.add_argument("--train_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "dataset/processed/training")),
                        help="Path to training data directory")
    parser.add_argument("--test_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "dataset/processed/testing")),
                        help="Path to testing data directory")
    parser.add_argument("--feature_type", type=str, choices=["hog", "pixels"],
                        default="hog", help="Feature extraction method")
    parser.add_argument("--save_path", type=str,
                        default=str(Path(PROJECT_ROOT / "checkpoints/naive_bayes/model.pkl")),
                        help="Path to save the trained model")
    parser.add_argument("--img_size", type=int, nargs=2, default=[64, 64],
                        help="Image size for feature extraction (height width)")

    args = parser.parse_args()

    print("=== Naive Bayes Training ===")
    print(f"Feature type: {args.feature_type}")
    print(f"Image size: {args.img_size}")
    print(f"Train dir: {args.train_dir}")
    print(f"Test dir: {args.test_dir}")
    print(f"Save path: {args.save_path}")
    print()

    # Train the model
    model, results = train_naive_bayes(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        feature_type=args.feature_type,
        save_path=args.save_path
    )

    print("\n=== Training Complete ===")
    print(f"Model saved to: {args.save_path}")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()