"""
The :mod:`skactiveml.pool` module implements query strategies for pool-based
active learning.
"""

from ._probal import McPAL
from ._random import RandomSampler
from ._uncertainty import UncertaintySampling, expected_average_precision
from ._epistemic_uncertainty import EpistemicUncertainty
from ._qbc import QBC, average_kl_divergence, vote_entropy
from ._expected_error import ExpectedErrorReduction
from ._four_ds import FourDS

__all__ = ['RandomSampler', 'McPAL', 'UncertaintySampling',
           'EpistemicUncertainty', 'ExpectedErrorReduction', 'QBC', 'FourDS']
