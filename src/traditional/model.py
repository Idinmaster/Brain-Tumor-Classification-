"""
Traditional ML Pipeline Implementation
=====================================
Gaussian filtering → HOG features → PCA → LDA → SVM
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage import io, color, filters, transform
from skimage.feature import hog
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.modern.dataset import CLASSES
from .utils import TraditionalConfig


class TraditionalPipeline:
    """Traditional ML pipeline: Gaussian → HOG → PCA → LDA → SVM"""

    def __init__(self, config: TraditionalConfig = None):
        """
        Initialize the traditional ML pipeline.

        Args:
            config: Configuration object with pipeline parameters
        """
        self.config = config or TraditionalConfig()
        self.classes_ = CLASSES

        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.pca_variance_ratio)
        self.lda = LDA(n_components=self.config.lda_n_components)
        self.svm = SVC(
            kernel=self.config.svm_kernel,
            C=self.config.svm_C,
            gamma=self.config.svm_gamma,
            random_state=42
        )

        # Track if components are fitted
        self.is_fitted = False

    def apply_gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filtering to the image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Apply Gaussian filter
        filtered = filters.gaussian(
            image,
            sigma=self.config.gaussian_sigma,
            mode=self.config.gaussian_padding,
            truncate=3.0
        )

        return filtered

    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from the image."""
        # Resize image to consistent size
        image = transform.resize(
            image,
            self.config.hog_image_size,
            anti_aliasing=True
        )

        # Extract HOG features
        features = hog(
            image,
            orientations=self.config.hog_num_bins,
            pixels_per_cell=self.config.hog_cell_size,
            cells_per_block=self.config.hog_block_size,
            block_norm='L2-Hys',
            feature_vector=True
        )

        return features

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Complete preprocessing pipeline for a single image."""
        # Load image
        image = io.imread(image_path)

        # Apply Gaussian filtering
        filtered = self.apply_gaussian_filter(image)

        # Extract HOG features
        features = self.extract_hog_features(filtered)

        return features

    def extract_features_from_directory(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all images in a directory."""
        data_dir = Path(data_dir)
        features_list = []
        labels_list = []

        print(f"Extracting features from {data_dir}...")

        for class_idx, class_name in enumerate(self.classes_):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue

            image_paths = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))

            for img_path in tqdm(image_paths, desc=f"Processing {class_name}"):
                try:
                    features = self.preprocess_image(str(img_path))
                    features_list.append(features)
                    labels_list.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

        X = np.array(features_list)
        y = np.array(labels_list)

        print(f"Extracted features: {X.shape}, Labels: {y.shape}")
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the complete pipeline."""
        print("Training traditional ML pipeline...")

        # Step 1: Standardize features
        print("Step 1: Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: PCA for dimensionality reduction
        print("Step 2: Applying PCA...")
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")

        # Step 3: LDA for further reduction to 3 dimensions
        print("Step 3: Applying LDA...")
        X_lda = self.lda.fit_transform(X_pca, y)
        print(f"LDA reduced dimensions to {X_lda.shape[1]}")

        # Step 4: Train SVM
        print("Step 4: Training SVM...")
        self.svm.fit(X_lda, y)

        self.is_fitted = True
        print("Pipeline training completed!")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features through the pipeline (without SVM)."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming")

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_lda = self.lda.transform(X_pca)
        return X_lda

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before predicting")

        X_transformed = self.transform(X)
        return self.svm.predict(X_transformed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before predicting")

        X_transformed = self.transform(X)
        return self.svm.predict_proba(X_transformed)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))

    def save_model(self, filepath: str) -> None:
        """Save the trained pipeline."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")

        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'pca': self.pca,
            'lda': self.lda,
            'svm': self.svm,
            'classes': self.classes_,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Pipeline saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained pipeline."""
        model_data = joblib.load(filepath)
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.lda = model_data['lda']
        self.svm = model_data['svm']
        self.classes_ = model_data['classes']
        self.is_fitted = model_data['is_fitted']
        print(f"Pipeline loaded from {filepath}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                save_plot: Optional[str] = None) -> dict:
        """Evaluate the pipeline and return detailed metrics."""
        print("Evaluating traditional ML pipeline...")

        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(".4f")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        if save_plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.classes_, yticklabels=self.classes_)
            plt.title('Confusion Matrix - Traditional ML Pipeline')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {save_plot}")

        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.classes_, output_dict=True
            )
        }

        return results


def create_traditional_pipeline(config: TraditionalConfig = None) -> TraditionalPipeline:
    """Factory function to create a traditional ML pipeline."""
    return TraditionalPipeline(config)


def train_traditional_pipeline(train_dir: str, test_dir: str,
                              config: TraditionalConfig = None,
                              save_path: Optional[str] = None,
                              save_plot: Optional[str] = None) -> Tuple[TraditionalPipeline, dict]:
    """
    Train and evaluate a traditional ML pipeline.

    Args:
        train_dir: Path to training data directory
        test_dir: Path to testing data directory
        config: Pipeline configuration
        save_path: Optional path to save the trained pipeline
        save_plot: Optional path to save confusion matrix plot

    Returns:
        Trained pipeline and evaluation results
    """
    # Create pipeline
    pipeline = create_traditional_pipeline(config)

    # Extract features
    print("=== Feature Extraction ===")
    X_train, y_train = pipeline.extract_features_from_directory(train_dir)
    X_test, y_test = pipeline.extract_features_from_directory(test_dir)

    # Train pipeline
    print("\n=== Training Pipeline ===")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluation ===")
    results = pipeline.evaluate(X_test, y_test, save_plot)

    # Save pipeline if requested
    if save_path:
        pipeline.save_model(save_path)

    return pipeline, results


if __name__ == "__main__":
    # Quick test
    pipeline = create_traditional_pipeline()
    print("Traditional ML pipeline created successfully!")
    print(f"Configuration: Gaussian kernel={pipeline.config.gaussian_kernel_size}, "
          f"HOG bins={pipeline.config.hog_num_bins}, "
          f"PCA variance={pipeline.config.pca_variance_ratio}")