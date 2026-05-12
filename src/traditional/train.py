"""
Traditional ML Pipeline Training Script
=======================================
Train the traditional pipeline: Gaussian → HOG → PCA → LDA → SVM
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.traditional.model import train_traditional_pipeline
from src.traditional.utils import TraditionalConfig


def main():
    parser = argparse.ArgumentParser(description="Train traditional ML pipeline")
    parser.add_argument("--train_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "dataset/raw/training")),
                        help="Path to training data directory")
    parser.add_argument("--test_dir", type=str,
                        default=str(Path(PROJECT_ROOT / "dataset/raw/testing")),
                        help="Path to testing data directory")
    parser.add_argument("--save_path", type=str,
                        default=str(Path(PROJECT_ROOT / "checkpoints/traditional/pipeline.pkl")),
                        help="Path to save the trained pipeline")
    parser.add_argument("--save_plot", type=str,
                        default=str(Path(PROJECT_ROOT / "checkpoints/traditional/confusion_matrix.png")),
                        help="Path to save confusion matrix plot")

    # Pipeline parameters
    parser.add_argument("--gaussian_kernel_size", type=int, default=5,
                        help="Gaussian filter kernel size (odd number)")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0,
                        help="Gaussian filter sigma")
    parser.add_argument("--gaussian_padding", type=str, default="reflect",
                        choices=["reflect", "constant", "nearest", "mirror", "wrap"],
                        help="Gaussian filter boundary mode")
    parser.add_argument("--hog_cell_size", type=int, nargs=2, default=[8, 8],
                        help="HOG cell size (height width)")
    parser.add_argument("--hog_num_bins", type=int, default=9,
                        help="Number of HOG orientation bins")
    parser.add_argument("--pca_variance", type=float, default=0.95,
                        help="PCA variance ratio to retain")
    parser.add_argument("--svm_kernel", type=str, default="rbf",
                        choices=["linear", "poly", "rbf", "sigmoid"],
                        help="SVM kernel type")
    parser.add_argument("--svm_C", type=float, default=1.0,
                        help="SVM regularization parameter")

    args = parser.parse_args()

    print("=== Traditional ML Pipeline Training ===")
    print(f"Train dir: {args.train_dir}")
    print(f"Test dir: {args.test_dir}")
    print(f"Save path: {args.save_path}")
    print(f"Save plot: {args.save_plot}")
    print()

    # Create configuration
    config = TraditionalConfig(
        gaussian_kernel_size=args.gaussian_kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_padding=args.gaussian_padding,
        hog_cell_size=tuple(args.hog_cell_size),
        hog_num_bins=args.hog_num_bins,
        pca_variance_ratio=args.pca_variance,
        svm_kernel=args.svm_kernel,
        svm_C=args.svm_C
    )

    print("Configuration:")
    print(f"  Gaussian: kernel={config.gaussian_kernel_size}, sigma={config.gaussian_sigma}, padding={config.gaussian_padding}")
    print(f"  HOG: cell={config.hog_cell_size}, bins={config.hog_num_bins}")
    print(f"  PCA: variance={config.pca_variance_ratio}")
    print(f"  SVM: kernel={config.svm_kernel}, C={config.svm_C}")
    print()

    # Train the pipeline
    pipeline, results = train_traditional_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        config=config,
        save_path=args.save_path,
        save_plot=args.save_plot
    )

    print("\n=== Training Complete ===")
    print(f"Pipeline saved to: {args.save_path}")
    print(f"Confusion matrix saved to: {args.save_plot}")
    print(".4f")


if __name__ == "__main__":
    main()