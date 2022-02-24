"""
The :mod:`skactiveml.classifier` module.
"""
from ._cmm import CMM
from ._pwc import PWC
from ._wrapper import SklearnClassifier

__all__ = ['multi', 'PWC', 'CMM', 'SklearnClassifier']