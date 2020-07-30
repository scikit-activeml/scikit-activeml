"""
The :mod:`skactiveml.pool` module implements query strategies for pool-based active learning.
"""

from ._probal import McPAL, XPAL
from ._random import RandomSampler
from ._uncertainty import UncertaintySampling

__all__ = ['RandomSampler', 'McPAL', 'XPAL', 'UncertaintySampling']
