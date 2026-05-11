"""Utility classes for modern training configuration."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "dataset/raw/training"
DEFAULT_TEST_DIR = PROJECT_ROOT / "dataset/raw/testing"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CNN_ARCHITECTURES = ("simple_cnn", "resnet18", "resnet50", "efficientnet_b0")
TRANSFORMER_ARCHITECTURES = ("mini_vit", "vit_b16", "swin_t")
SUPPORTED_MODELS = ("cnn", "transformer")


@dataclass
class ModelConfig:
    model: str = "transformer"
    arch: str = "mini_vit"
    pretrained: bool = True
    epochs: int = 30
    warmup_epochs: int = 5
    batch_size: int = 32
    img_size: int = 224
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    num_workers: int = 4
    val_fraction: float = 1 / 6
    train_dir: str = field(default_factory=lambda: str(DEFAULT_TRAIN_DIR))
    test_dir: str = field(default_factory=lambda: str(DEFAULT_TEST_DIR))
    save_dir: str | None = None

    def __post_init__(self) -> None:
        if self.save_dir is None:
            self.save_dir = str(DEFAULT_CHECKPOINT_DIR / self.model)

        self.train_dir = str(Path(self.train_dir))
        self.test_dir = str(Path(self.test_dir))
        self.save_dir = str(Path(self.save_dir))

        self.validate()

    def validate(self) -> None:
        model = self.model.lower()
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{self.model}'. Choose from {SUPPORTED_MODELS}."
            )

        if model == "cnn" and self.arch not in CNN_ARCHITECTURES:
            raise ValueError(
                f"Unknown CNN architecture '{self.arch}'. "
                f"Choose from {CNN_ARCHITECTURES}."
            )
        if model == "transformer" and self.arch not in TRANSFORMER_ARCHITECTURES:
            raise ValueError(
                f"Unknown transformer architecture '{self.arch}'. "
                f"Choose from {TRANSFORMER_ARCHITECTURES}."
            )

        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.img_size <= 0:
            raise ValueError("img_size must be a positive integer")
        if self.lr <= 0.0:
            raise ValueError("lr must be a positive float")
        if self.min_lr < 0.0:
            raise ValueError("min_lr must be non-negative")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.label_smoothing < 0.0:
            raise ValueError("label_smoothing must be non-negative")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path | str | None = None) -> Path:
        if filepath is None:
            filepath = Path(self.save_dir) / "config.json"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

        return path

    @classmethod
    def load(cls, filepath: Path | str) -> "ModelConfig":
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)
