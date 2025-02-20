"""
The :mod:`skactiveml.stream` package implements query strategies for
stream-based active learning.
"""

from ._stream_baselines import StreamRandomSampling, PeriodicSampling
from ._stream_probabilistic_al import StreamProbabilisticAL
from ._uncertainty_zliobaite import (
    FixedUncertainty,
    VariableUncertainty,
    Split,
    RandomVariableUncertainty,
)
from ._density_uncertainty import (
    StreamDensityBasedAL,
    CognitiveDualQueryStrategy,
    CognitiveDualQueryStrategyRan,
    CognitiveDualQueryStrategyRanVarUn,
    CognitiveDualQueryStrategyVarUn,
    CognitiveDualQueryStrategyFixUn,
)

__all__ = [
    "budgetmanager",
    "StreamRandomSampling",
    "PeriodicSampling",
    "FixedUncertainty",
    "VariableUncertainty",
    "Split",
    "StreamProbabilisticAL",
    "RandomVariableUncertainty",
    "StreamDensityBasedAL",
    "CognitiveDualQueryStrategy",
    "CognitiveDualQueryStrategyRan",
    "CognitiveDualQueryStrategyRanVarUn",
    "CognitiveDualQueryStrategyVarUn",
    "CognitiveDualQueryStrategyFixUn",
]
