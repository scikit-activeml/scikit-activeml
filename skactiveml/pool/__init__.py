"""
The :mod:`skactiveml.pool` package implements query strategies for
pool-based active learning.
"""

from . import multiannotator
from . import utils
from ._cost_embedding_al import CostEmbeddingAL
from ._discriminative_al import DiscriminativeAL
from ._epistemic_uncertainty_sampling import EpistemicUncertaintySampling
from ._expected_error_reduction import MonteCarloEER, ValueOfInformationEER
from ._expected_model_change_maximization import (
    ExpectedModelChangeMaximization,
)
from ._expected_model_output_change import ExpectedModelOutputChange
from ._expected_model_variance import ExpectedModelVarianceReduction
from ._four_ds import FourDs
from ._greedy_sampling import GreedySamplingX, GreedySamplingTarget
from ._information_gain_maximization import (
    KLDivergenceMaximization,
)
from ._probabilistic_al import ProbabilisticAL, cost_reduction
from ._query_by_committee import (
    QueryByCommittee,
    average_kl_divergence,
    vote_entropy,
)
from ._quire import Quire
from ._random_sampling import RandomSampling
from ._uncertainty_sampling import (
    UncertaintySampling,
    uncertainty_scores,
    expected_average_precision,
)

__all__ = [
    "multiannotator",
    "utils",
    "RandomSampling",
    "ProbabilisticAL",
    "cost_reduction",
    "UncertaintySampling",
    "uncertainty_scores",
    "expected_average_precision",
    "EpistemicUncertaintySampling",
    "MonteCarloEER",
    "ValueOfInformationEER",
    "QueryByCommittee",
    "Quire",
    "average_kl_divergence",
    "vote_entropy",
    "FourDs",
    "CostEmbeddingAL",
    "ExpectedModelChangeMaximization",
    "ExpectedModelOutputChange",
    "ExpectedModelVarianceReduction",
    "KLDivergenceMaximization",
    "GreedySamplingX",
    "GreedySamplingTarget",
    "DiscriminativeAL",
]
