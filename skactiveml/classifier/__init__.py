"""
The :mod:`skactiveml.classifier` module.
"""

from ._mixture_model_classifier import MixtureModelClassifier
from ._parzen_window_classifier import ParzenWindowClassifier
from ._evidential_knn_classifier import EKNN
from ._wrapper import SklearnClassifier, SlidingWindowClassifier

__all__ = [
    "multiannotator",
    "ParzenWindowClassifier",
    "MixtureModelClassifier",
    "SklearnClassifier",
    "SlidingWindowClassifier",
    "EKNN"
]
