"""
The :mod:`skactiveml.regressor` module.
"""

from skactiveml.regressor._nwr import NWR
from skactiveml.regressor._wrapper import SklearnRegressor, SklearnConditionalEstimator

__all__ = [
    "estimator",
    "NWR",
    "SklearnRegressor",
    "SklearnConditionalEstimator",
]
