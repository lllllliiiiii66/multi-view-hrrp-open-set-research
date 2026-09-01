"""Shared neural components for the frozen baseline chain."""

from .cnn1d import HRRPClassifier1D, SharedHRRPEncoder1D

__all__ = ["HRRPClassifier1D", "SharedHRRPEncoder1D"]
from .cnn1d import HRRPClassifier1D, SharedHRRPEncoder1D
from .sets import DeepSetsClassifier, SetTransformerClassifier

__all__ = [
    "DeepSetsClassifier",
    "HRRPClassifier1D",
    "SetTransformerClassifier",
    "SharedHRRPEncoder1D",
]
