"""
The :mod:`skactiveml.classifier` module.
"""

from ._mixture_model_classifier import MixtureModelClassifier
from ._parzen_window_classifier import ParzenWindowClassifier
from ._wrapper import SklearnClassifier, SlidingWindowClassifier

__all__ = [
    "multiannotator",
    "ParzenWindowClassifier",
    "MixtureModelClassifier",
    "SklearnClassifier",
    "SlidingWindowClassifier",
]

try:
    from ._wrapper import SkorchClassifier  # noqa: F401

    __all__ += ["SkorchClassifier"]
except ImportError:  # pragma: no cover
    pass
