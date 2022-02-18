"""
The :mod:`skactiveml.pool` package implements query strategies for
pool-based active learning.
"""

from . import multi
from . import utils
from ._alce import ALCE
from ._epistemic_uncertainty import EpistemicUncertainty
from ._expected_error import MonteCarloEER, ValueOfInformationEER
from ._four_ds import FourDS
from ._probal import McPAL, cost_reduction
from ._qbc import QBC, average_kl_divergence, vote_entropy
from ._random import RandomSampler
from ._uncertainty import UncertaintySampling, uncertainty_scores, \
    expected_average_precision

__all__ = ['multi', 'utils', 'RandomSampler', 'McPAL', 'cost_reduction',
           'UncertaintySampling', 'uncertainty_scores',
           'expected_average_precision', 'EpistemicUncertainty',
           'MonteCarloEER', 'ValueOfInformationEER', 'QBC',
           'average_kl_divergence', 'vote_entropy', 'FourDS', 'ALCE']
