"""
The :mod:`skactiveml.stream.budget_manager` module implements budget managers for stream-based active learning.
"""

from .base import BudgetManager
from ._fixed_budget import FixedBudget

__all__ = ['BudgetManager', 'FixedBudget']
