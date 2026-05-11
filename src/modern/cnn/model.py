"""
CNN Model for Brain Tumor Classification
=========================================
Supports:
  - Custom CNN (SimpleCNN)
  - Transfer Learning with ResNet18 / ResNet50 / EfficientNet-B0

Classes: glioma, meningioma, notumor, pituitary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Custom CNN Architecture
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Convolution → BatchNorm → ReLU → optional MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    Custom CNN for 4-class brain tumor classification.

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes)

    Spatial flow (224 input):
        Block1: 224 → 112   (×32 channels)
        Block2: 112 →  56   (×64 channels)
        Block3:  56 →  28   (×128 channels)
        Block4:  28 →  14   (×256 channels)
    Flatten: 256 × 14 × 14 = 50176
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),          # robust to input sizes
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Transfer-Learning Wrappers
# ---------------------------------------------------------------------------

def _replace_head(model: nn.Module, in_features: int, num_classes: int) -> nn.Module:
    """Replace the final FC layer with a custom head."""
    head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return head


def get_resnet18(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with a custom classification head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = _replace_head(model, in_features, num_classes)
    return model


def get_resnet50(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with a custom classification head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = _replace_head(model, in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with a custom classification head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = _replace_head(model, in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SUPPORTED_ARCHITECTURES = {
    "simple_cnn":      lambda nc, pt: SimpleCNN(num_classes=nc),
    "resnet18":        get_resnet18,
    "resnet50":        get_resnet50,
    "efficientnet_b0": get_efficientnet_b0,
}


def get_cnn_model(
    architecture: str = "resnet18",
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function for CNN models.

    Args:
        architecture: One of 'simple_cnn', 'resnet18', 'resnet50', 'efficientnet_b0'.
        num_classes:  Number of output classes (default 4).
        pretrained:   Use ImageNet pretrained weights (ignored for simple_cnn).

    Returns:
        nn.Module ready for training.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(SUPPORTED_ARCHITECTURES.keys())}"
        )
    return SUPPORTED_ARCHITECTURES[architecture](num_classes, pretrained)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 224, 224).to(device)

    for arch in SUPPORTED_ARCHITECTURES:
        model = get_cnn_model(arch, num_classes=4, pretrained=False).to(device)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{arch:20s}]  output={out.shape}  trainable params={params:,}")
