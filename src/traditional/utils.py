"""
Traditional ML Pipeline Configuration
====================================
Configuration parameters for the traditional machine learning pipeline:
Gaussian filtering → HOG features → PCA → LDA → SVM
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TraditionalConfig:
    """Configuration for traditional ML pipeline."""

    # Gaussian filtering parameters
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    gaussian_padding: str = "reflect"  # "reflect", "constant", "nearest", "mirror", "wrap"

    # HOG feature parameters
    hog_cell_size: tuple[int, int] = (8, 8)
    hog_block_size: tuple[int, int] = (2, 2)
    hog_num_bins: int = 9
    hog_image_size: tuple[int, int] = (128, 128)

    # PCA parameters
    pca_variance_ratio: float = 0.95

    # LDA parameters (fixed to 3 dimensions for 4 classes)
    lda_n_components: int = 3

    # SVM parameters
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str = "scale"

    # Data paths
    train_dir: str = str(PROJECT_ROOT / "dataset/processed/training")
    test_dir: str = str(PROJECT_ROOT / "dataset/processed/testing")
    save_dir: str = str(PROJECT_ROOT / "checkpoints/traditional")

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate Gaussian parameters
        if self.gaussian_kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel size must be odd")
        if self.gaussian_kernel_size < 1:
            raise ValueError("Gaussian kernel size must be positive")
        if self.gaussian_sigma <= 0:
            raise ValueError("Gaussian sigma must be positive")
        if self.gaussian_padding not in ["reflect", "constant", "nearest", "mirror", "wrap"]:
            raise ValueError("Gaussian padding must be one of: reflect, constant, nearest, mirror, wrap")

        # Validate HOG parameters
        if self.hog_cell_size[0] <= 0 or self.hog_cell_size[1] <= 0:
            raise ValueError("HOG cell size must be positive")
        if self.hog_block_size[0] <= 0 or self.hog_block_size[1] <= 0:
            raise ValueError("HOG block size must be positive")
        if self.hog_num_bins <= 0:
            raise ValueError("HOG number of bins must be positive")
        if self.hog_image_size[0] <= 0 or self.hog_image_size[1] <= 0:
            raise ValueError("HOG image size must be positive")

        # Validate PCA parameters
        if not 0 < self.pca_variance_ratio <= 1:
            raise ValueError("PCA variance ratio must be between 0 and 1")

        # Validate LDA parameters
        if self.lda_n_components <= 0:
            raise ValueError("LDA n_components must be positive")

        # Validate SVM parameters
        if self.svm_kernel not in ["linear", "poly", "rbf", "sigmoid"]:
            raise ValueError("SVM kernel must be one of: linear, poly, rbf, sigmoid")
        if self.svm_C <= 0:
            raise ValueError("SVM C must be positive")
        if self.svm_gamma not in ["scale", "auto"] and not isinstance(self.svm_gamma, (int, float)):
            raise ValueError("SVM gamma must be 'scale', 'auto', or a positive number")

        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = TraditionalConfig()