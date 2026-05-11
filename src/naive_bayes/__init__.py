"""
Naive Bayes Baseline Model
==========================
Traditional machine learning baseline using scikit-learn's Naive Bayes
with feature extraction from images.
"""

from .model import get_naive_bayes_model

__all__ = ["get_naive_bayes_model"]