from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_scalar

class BudgetManager(ABC, BaseEstimator):
    
    def __init__(self, budget):
        check_scalar(budget, 'budget', np.float, min_val=0.0, max_val=1.0)
        self.budget = budget
    
    def is_budget_left(self):
        return NotImplemented
    
    def sample(self, utilities, return_budget_left=False, simulate=False, **kwargs):
        return NotImplemented