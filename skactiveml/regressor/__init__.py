"""
The :mod:`skactiveml.regressor` module.
"""

from skactiveml.regressor._nic_kernel_regressor import (
    NICKernelRegressor,
    NadarayaWatsonRegressor,
)
from skactiveml.regressor._wrapper import (
    SklearnRegressor,
    SklearnNormalRegressor,
)

__all__ = [
    "NICKernelRegressor",
    "NadarayaWatsonRegressor",
    "SklearnRegressor",
    "SklearnNormalRegressor",
]
