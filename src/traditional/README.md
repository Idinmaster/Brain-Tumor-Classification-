# Traditional ML Pipeline

This directory contains a traditional machine learning pipeline for brain tumor classification:

**Pipeline:** Gaussian Filtering → HOG Features → PCA → LDA → SVM

## Features

- **Gaussian Filtering**: Smooths images to reduce noise
- **HOG Features**: Extracts gradient-based features robust to illumination changes
- **PCA**: Reduces dimensionality while retaining 95% of variance
- **LDA**: Further reduces to 3 dimensions optimized for classification
- **SVM**: Final classification with configurable kernel

## Configuration

All parameters are configurable in `utils.py`:

```python
config = TraditionalConfig(
    # Gaussian filtering
    gaussian_kernel_size=5,
    gaussian_sigma=1.0,
    gaussian_padding="reflect",  # "reflect", "constant", "nearest", "mirror", "wrap"

    # HOG features
    hog_cell_size=(8, 8),
    hog_num_bins=9,

    # Dimensionality reduction
    pca_variance_ratio=0.95,  # Retain 95% variance
    lda_n_components=3,       # Reduce to 3 dimensions

    # SVM
    svm_kernel="rbf",
    svm_C=1.0
)
```

## Usage

### Training

```bash
# Train with default parameters
python src/traditional/train.py

# Custom configuration
python src/traditional/train.py \
    --gaussian_kernel_size 7 \
    --hog_num_bins 12 \
    --pca_variance 0.99 \
    --svm_kernel linear
```

### Python API

```python
from src.traditional.model import train_traditional_pipeline
from src.traditional.utils import TraditionalConfig

# Custom config
config = TraditionalConfig(
    gaussian_kernel_size=7,
    hog_num_bins=12,
    pca_variance_ratio=0.99
)

# Train and evaluate
pipeline, results = train_traditional_pipeline(
    train_dir="dataset/processed/training",
    test_dir="dataset/processed/testing",
    config=config,
    save_path="checkpoints/traditional/pipeline.pkl",
    save_plot="checkpoints/traditional/cm.png"
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

## Output

The pipeline generates:
- Trained model (`pipeline.pkl`)
- Confusion matrix plot (`confusion_matrix.png`)
- Detailed classification metrics

## Performance

Expected performance range:
- **Accuracy**: 75-85% (depends on parameters and data)
- **Training time**: 2-5 minutes (feature extraction + training)
- **Inference time**: Fast (< 1 second per image)

## Comparison to Deep Learning

| Method | Accuracy | Training Time | Inference Speed |
|--------|----------|---------------|-----------------|
| Traditional ML | 75-85% | Fast | Very Fast |
| CNN | 85-95% | Slow | Fast |
| Transformer | 88-96% | Very Slow | Moderate |

Use this as a strong baseline to compare against deep learning models!