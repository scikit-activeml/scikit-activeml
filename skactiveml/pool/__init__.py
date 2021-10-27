"""
The :mod:`skactiveml.pool` package implements query strategies for
pool-based active learning.
"""

from . import multi
from ._alce import ALCE
from ._epistemic_uncertainty import EpistemicUncertainty
from ._expected_error import ExpectedErrorReduction, expected_error_reduction
from ._four_ds import FourDS
from ._probal import McPAL, cost_reduction
from ._qbc import QBC, average_kl_divergence, vote_entropy
from ._random import RandomSampler
from ._uncertainty import UncertaintySampling, uncertainty_scores, \
    expected_average_precision

__all__ = ['multi', 'RandomSampler', 'McPAL', 'cost_reduction',
           'UncertaintySampling', 'uncertainty_scores',
           'expected_average_precision', 'EpistemicUncertainty',
           'ExpectedErrorReduction', 'expected_error_reduction', 'QBC',
           'average_kl_divergence', 'vote_entropy', 'FourDS', 'ALCE']
