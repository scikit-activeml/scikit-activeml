"""
The :mod:`skactiveml.visualization` module includes various tools for
visualization.
"""
from ._feature_space import plot_utility, plot_decision_boundary

__all__ = ['plot_utility', 'plot_decision_boundary', 'plot_ma_current_state',
           'plot_ma_utility', 'plot_ma_decision_boundary']

from .multi import plot_ma_current_state, plot_ma_utility, \
    plot_ma_decision_boundary
