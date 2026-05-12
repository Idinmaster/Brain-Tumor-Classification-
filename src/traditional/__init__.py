"""
Traditional Machine Learning Pipeline
====================================
Gaussian filtering → HOG features → PCA → LDA → SVM
"""

from .model import TraditionalPipeline
from .utils import TraditionalConfig, default_config

__all__ = ["TraditionalPipeline", "TraditionalConfig", "default_config"]