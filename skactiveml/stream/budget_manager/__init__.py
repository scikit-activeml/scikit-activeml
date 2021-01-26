"""
The :mod:`skactiveml.stream.budget_manager` module implements budget managers,
which are capable of modeling the budget constraints in stream-based active
learning settings.
"""

from .base import BudgetManager
from ._fixed_budget import FixedBudget
from ._estimated_budget import (
    EstimatedBudget,
    FixedUncertaintyBudget,
    VarUncertaintyBudget,
    SplitBudget,
)

__all__ = [
    "BudgetManager",
    "FixedBudget",
    "EstimatedBudget",
    "FixedUncertaintyBudget",
    "VarUncertaintyBudget",
    "SplitBudget",
]
