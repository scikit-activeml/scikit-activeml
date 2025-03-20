"""
The :mod:`skactiveml.pool` package implements query strategies for
pool-based active learning.
"""

from . import multiannotator
from . import utils
from ._bald import GreedyBALD, BatchBALD, batch_bald
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
from ._information_gain_maximization import KLDivergenceMaximization
from ._probabilistic_al import ProbabilisticAL, cost_reduction
from ._query_by_committee import (
    QueryByCommittee,
    average_kl_divergence,
    vote_entropy,
    variation_ratios,
)
from ._quire import Quire
from ._random_sampling import RandomSampling
from ._regression_tree_based_al import RegressionTreeBasedAL
from ._uncertainty_sampling import (
    UncertaintySampling,
    uncertainty_scores,
    expected_average_precision,
)
from ._core_set import CoreSet, k_greedy_center
from ._typi_clust import TypiClust
from ._badge import Badge
from ._prob_cover import ProbCover
from ._contrastive_al import ContrastiveAL
from ._clue import Clue
from ._drop_query import DropQuery
from ._wrapper import SubSamplingWrapper, ParallelUtilityEstimationWrapper
from ._falcun import Falcun

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
    "variation_ratios",
    "FourDs",
    "CostEmbeddingAL",
    "ExpectedModelChangeMaximization",
    "ExpectedModelOutputChange",
    "ExpectedModelVarianceReduction",
    "KLDivergenceMaximization",
    "GreedySamplingX",
    "GreedySamplingTarget",
    "DiscriminativeAL",
    "BatchBALD",
    "Clue",
    "DropQuery",
    "batch_bald",
    "CoreSet",
    "k_greedy_center",
    "TypiClust",
    "Badge",
    "ProbCover",
    "ContrastiveAL",
    "GreedyBALD",
    "RegressionTreeBasedAL",
    "SubSamplingWrapper",
    "ParallelUtilityEstimationWrapper",
    "Falcun",
]
