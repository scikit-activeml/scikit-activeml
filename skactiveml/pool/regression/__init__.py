"""
The :mod:`skactiveml.pool.regression` package implements query strategies for
regression and pool-based active learning.
"""

from skactiveml.pool import EpistemicUncertaintySampling
from skactiveml.pool.regression._expected_model_change import ExpectedModelChange
from skactiveml.pool.regression._expected_model_output_change import (
    ExpectedModelOutputChange,
)
from skactiveml.pool.regression._expected_model_variance import (
    ExpectedModelVarianceMinimization,
)
from skactiveml.pool.regression._greedy_sampling_x import GSx
from skactiveml.pool.regression._greedy_sampling_y import GSy
from skactiveml.pool.regression._kl_divergence_maximization import (
    KullbackLeiblerDivergenceMaximization,
    cross_entropy,
)
from skactiveml.pool.regression._mutual_information_maximization import (
    MutualInformationGainMaximization,
)
from skactiveml.pool.regression._query_by_committee import QueryByCommittee
from skactiveml.pool.regression._representativeness_and_diversity import (
    RD,
)

__all__ = [
    "utils",
    "ExpectedModelChange",
    "ExpectedModelOutputChange",
    "ExpectedModelVarianceMinimization",
    "GSx",
    "GSy",
    "KullbackLeiblerDivergenceMaximization",
    "cross_entropy",
    "MutualInformationGainMaximization",
    "QueryByCommittee",
    "EpistemicUncertaintySampling",
    "RD",
]
