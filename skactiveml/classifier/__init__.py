"""
The :mod:`skactiveml.pool` module implements query strategies for pool-based active learning.
"""

from ._meta import IgnUnlabeledClassifier
from ._pwc import PWC

__all__ = ['IgnUnlabeledClassifier', 'PWC']
