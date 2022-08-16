"""
The :mod:`skactiveml.classifier` module.
"""
from ._mixture_model_classifier import MixtureModelClassifier
from ._parzen_window_classifier import (
    ParzenWindowClassifier,
    ALLOWED_MEAN_KERNEL_METRICS,
)
from ._wrapper import SklearnClassifier, SlidingWindowClassifier

__all__ = [
    "multiannotator",
    "ParzenWindowClassifier",
    "MixtureModelClassifier",
    "SklearnClassifier",
    "SlidingWindowClassifier",
    "ALLOWED_MEAN_KERNEL_METRICS",
]
