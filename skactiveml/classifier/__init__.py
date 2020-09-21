"""
The :mod:`skactiveml.classifier` module # TODO.
"""

from ._pwc import PWC
from ._cmm import CMM
from ._wrapper import SklearnClassifier
from ._multi_annot_classifier import MultiAnnotClassifier

__all__ = ['PWC', 'CMM', 'SklearnClassifier', 'MultiAnnotClassifier']
