"""
The :mod:`skactiveml.regressor` module.
"""
from skactiveml.regressor._nichke import NICKernelRegressor
from skactiveml.regressor._nwr import NadarayaWatsonRegressor
from skactiveml.regressor._wrapper import (
    SklearnRegressor,
    SklearnTargetDistributionRegressor,
)

__all__ = [
    "NICKernelRegressor",
    "NadarayaWatsonRegressor",
    "SklearnRegressor",
    "SklearnTargetDistributionRegressor",
]
