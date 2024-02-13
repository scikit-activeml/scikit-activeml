"""
The :mod:`skactiveml.visualization` module includes various tools for
visualization.
"""

from ._feature_space import (
    plot_utilities,
    plot_decision_boundary,
    plot_contour_for_samples,
    plot_annotator_utilities,
    plot_stream_training_data,
    plot_stream_decision_boundary,
)
from ._misc import mesh

__all__ = [
    "plot_utilities",
    "plot_decision_boundary",
    "plot_contour_for_samples",
    "plot_annotator_utilities",
    "plot_stream_training_data",
    "plot_stream_decision_boundary",
    "mesh",
]
