"""
The :mod:`skactiveml.regressor` module.
"""
from skactiveml.regressor._nic_kernel_regressor import NICKernelRegressor
from skactiveml.regressor._nadaraya_watson_regressor import NadarayaWatsonRegressor
from skactiveml.regressor._wrapper import (
    SklearnRegressor,
    SklearnProbabilisticRegressor,
)

__all__ = [
    "NICKernelRegressor",
    "NadarayaWatsonRegressor",
    "SklearnRegressor",
    "SklearnProbabilisticRegressor",
]
