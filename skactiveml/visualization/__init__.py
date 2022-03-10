"""
The :mod:`skactiveml.visualization` module includes various tools for
visualization.
"""
from ._feature_space import plot_utilities, plot_decision_boundary, \
    plot_contour_for_samples
from ._misc import mesh

__all__ = ['plot_utilities', 'plot_decision_boundary',
           'plot_contour_for_samples',
           'mesh']
