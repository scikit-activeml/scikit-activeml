"""
The :mod:`skactiveml.stream.budget_manager` module implements budget managers,
which are capable of modeling the budget constraints in stream-based active
learning settings.
"""

from .base import BudgetManager
from ._fixed_budget import FixedBudget

__all__ = ['BudgetManager', 'FixedBudget']
