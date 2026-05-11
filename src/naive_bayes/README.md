# Naive Bayes Baseline

This directory contains a traditional machine learning baseline using scikit-learn's Gaussian Naive Bayes classifier for brain tumor classification.

## Features

- **Feature Extraction**: HOG (Histogram of Oriented Gradients) or raw pixel features
- **Model**: Gaussian Naive Bayes classifier
- **Evaluation**: Accuracy and detailed classification report

## Usage

### Training

```bash
# Train with HOG features (recommended)
python src/naive_bayes/train.py --feature_type hog

# Train with raw pixel features
python src/naive_bayes/train.py --feature_type pixels

# Custom paths
python src/naive_bayes/train.py \
    --train_dir /path/to/training/data \
    --test_dir /path/to/testing/data \
    --save_path /path/to/save/model.pkl
```

### Python API

```python
from src.naive_bayes.model import train_naive_bayes

# Train and evaluate
model, results = train_naive_bayes(
    train_dir="dataset/processed/training",
    test_dir="dataset/processed/testing",
    feature_type="hog"
)

print(f"Test accuracy: {results['test_accuracy']:.4f}")
```

## Feature Types

### HOG Features (Recommended)
- Extracts gradient orientation histograms
- More robust to illumination changes
- Higher dimensional feature vectors
- Better performance for image classification

### Raw Pixel Features
- Simple flattened pixel values
- Faster to compute
- Sensitive to illumination and scale changes
- Lower performance but good for comparison

## Output

The model saves:
- Trained classifier (`model.pkl`)
- Feature extraction parameters
- Class labels

## Performance

Expected baseline performance:
- HOG features: ~70-80% accuracy
- Raw pixels: ~60-70% accuracy

Use this as a comparison point for deep learning models (CNN/Transformer).