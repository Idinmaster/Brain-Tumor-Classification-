"""
Naive Bayes Model Implementation
================================
Baseline model using scikit-learn's Gaussian Naive Bayes classifier
with feature extraction from images.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color
from skimage.feature import hog
from skimage.transform import resize
import joblib
from tqdm import tqdm

from src.modern.dataset import CLASSES


class NaiveBayesClassifier:
    """Naive Bayes classifier with image feature extraction."""

    def __init__(self, feature_type: str = "hog", img_size: Tuple[int, int] = (64, 64)):
        """
        Args:
            feature_type: "hog" for HOG features, "pixels" for raw pixel features
            img_size: Size to resize images to for feature extraction
        """
        self.feature_type = feature_type
        self.img_size = img_size
        self.model = GaussianNB()
        self.classes_ = CLASSES

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image."""
        # Load and preprocess image
        image = io.imread(image_path)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Resize image
        image = resize(image, self.img_size, anti_aliasing=True)

        if self.feature_type == "hog":
            # Extract HOG features
            features = hog(image,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys')
        elif self.feature_type == "pixels":
            # Use raw pixel values as features
            features = image.flatten()
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        return features

    def extract_features_from_directory(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all images in a directory."""
        data_dir = Path(data_dir)
        features_list = []
        labels_list = []

        print(f"Extracting {self.feature_type} features from {data_dir}...")

        for class_idx, class_name in enumerate(self.classes_):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue

            image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))

            for img_path in tqdm(image_paths, desc=f"Processing {class_name}"):
                try:
                    features = self.extract_features(str(img_path))
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
        """Train the Naive Bayes model."""
        print("Training Naive Bayes model...")
        self.model.fit(X, y)
        print("Training completed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'feature_type': self.feature_type,
            'img_size': self.img_size,
            'classes': self.classes_
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_type = model_data['feature_type']
        self.img_size = model_data['img_size']
        self.classes_ = model_data['classes']
        print(f"Model loaded from {filepath}")


def get_naive_bayes_model(feature_type: str = "hog", img_size: Tuple[int, int] = (64, 64)) -> NaiveBayesClassifier:
    """
    Factory function to create a Naive Bayes classifier.

    Args:
        feature_type: "hog" or "pixels"
        img_size: Image size for feature extraction

    Returns:
        Configured NaiveBayesClassifier instance
    """
    return NaiveBayesClassifier(feature_type=feature_type, img_size=img_size)


def train_naive_bayes(train_dir: str, test_dir: str, feature_type: str = "hog",
                     save_path: Optional[str] = None) -> Tuple[NaiveBayesClassifier, dict]:
    """
    Train and evaluate a Naive Bayes model.

    Args:
        train_dir: Path to training data directory
        test_dir: Path to testing data directory
        feature_type: Feature extraction method ("hog" or "pixels")
        save_path: Optional path to save the trained model

    Returns:
        Trained model and evaluation results
    """
    # Create and train model
    model = get_naive_bayes_model(feature_type=feature_type)

    # Extract features
    print("=== Feature Extraction ===")
    X_train, y_train = model.extract_features_from_directory(train_dir)
    X_test, y_test = model.extract_features_from_directory(test_dir)

    # Train model
    print("\n=== Training ===")
    model.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluation ===")
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(".4f")
    print(".4f")

    # Detailed classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Save model if requested
    if save_path:
        model.save_model(save_path)

    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_type': feature_type,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    return model, results


if __name__ == "__main__":
    # Quick test
    model = get_naive_bayes_model()
    print(f"Created Naive Bayes model with {model.feature_type} features")