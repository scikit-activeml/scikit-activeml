"""
The :mod:`skactiveml.pool.multiannotator` package implements multi annotator
pool-based active learning.
"""

from ._interval_estimation_threshold import (
    IntervalEstimationThreshold,
    IntervalEstimationAnnotModel,
)
from ._wrapper import SingleAnnotatorWrapper

__all__ = [
    "IntervalEstimationThreshold",
    "IntervalEstimationAnnotModel",
    "SingleAnnotatorWrapper",
]
