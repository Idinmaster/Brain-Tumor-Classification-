# Inference and Visualization

This directory contains tools for running inference with trained models and visualizing their predictions.

## Scripts

### `visualize_predictions.py`

Visualize examples of correct and incorrect predictions from a trained model.

**Usage:**
```bash
# Visualize transformer model predictions
python src/inference/visualize_predictions.py --model_dir checkpoints/transformer

# Visualize CNN model predictions
python src/inference/visualize_predictions.py --model_dir checkpoints/cnn

# Customize number of examples
python src/inference/visualize_predictions.py \
    --model_dir checkpoints/transformer \
    --num_display 5 \
    --num_examples 100
```

### `test_inference.py`

Quick test script that saves a visualization to a file instead of displaying it.

**Usage:**
```bash
# Test with default transformer model
python src/inference/test_inference.py

# Output: inference_test.png
```

**Parameters:**
- `--model_dir`: Path to model checkpoint directory (must contain `config.json` and `.pth` file)
- `--num_display`: Number of examples to display for each category (default: 3)
- `--num_examples`: Number of examples to collect for sampling (default: 50)

**Output:**
- Displays a matplotlib figure with 2 rows:
  - Top row: Correct predictions (green titles)
  - Bottom row: Incorrect predictions (red titles)
- Each image shows the true label and predicted label

## Supported Models

Works with any model saved in the `checkpoints/` directory that has:
- `config.json`: Model configuration file
- `*.pth`: PyTorch checkpoint file (automatically finds best model)

## Example Output

```
=== Model Inference Visualization ===
Model directory: checkpoints/transformer
Examples to collect: 50
Examples to display: 3

Loaded transformer model: mini_vit
Using device: cpu

[Dataset]  Train: 1600 images | Test: 1600 images
[Dataset]  Classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
Collecting prediction examples...
Found 50 correct and 50 incorrect examples

Displaying prediction examples...
```

The script will then show a visualization window with example images.