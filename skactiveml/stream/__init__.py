"""
The :mod:`skactiveml.pool` module implements query strategies for stream-based
active learning.
"""

from ._random import RandomSampler, PeriodicSampler

__all__ = ['RandomSampler', 'PeriodicSampler']
