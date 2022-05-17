"""
The :mod:`skactiveml.pool.regression` package implements query strategies for
pool-based active learning for regression.
"""

from skactiveml.pool.regression._expected_model_change import (
    ExpectedModelChange,
)
from skactiveml.pool.regression._expected_model_output_change import (
    ExpectedModelOutputChange,
)
from skactiveml.pool.regression._expected_model_variance import (
    ExpectedModelVarianceReduction,
)
from skactiveml.pool.regression._greedy_sampling import (
    GreedySamplingX,
    GreedySamplingY,
)
from skactiveml.pool.regression._information_gain_maximization import (
    MutualInformationGainMaximization,
    KLDivergenceMaximization,
    cross_entropy,
)
from skactiveml.pool.regression._query_by_committee import (
    QueryByCommittee,
)
from skactiveml.pool.regression._representativeness_and_diversity import (
    RepresentativenessDiversity,
)

__all__ = [
    "utils",
    "ExpectedModelChange",
    "ExpectedModelOutputChange",
    "ExpectedModelVarianceReduction",
    "KLDivergenceMaximization",
    "cross_entropy",
    "MutualInformationGainMaximization",
    "QueryByCommittee",
    "RepresentativenessDiversity",
    "GreedySamplingX",
    "GreedySamplingY",
]
