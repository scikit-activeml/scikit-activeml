import numpy as np

from sklearn.utils import check_array

from ..base import StreamBasedQueryStrategy
from .budget_manager import FixedBudget


class RandomSampler(StreamBasedQueryStrategy):
    def __init__(self, budget_manager, random_state=None):
        super().__init__(budget_manager=budget_manager, random_state=random_state)
    
    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)
        
        utilities = self.random_state.random_sample(len(X_cand))
        
        sampled_indices = self.budget_manager.sample(utilities, simulate=simulate)
        
        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices
        
        
class PeriodicSampler(StreamBasedQueryStrategy):
    def __init__(self, budget_manager, random_state=None):
        super().__init__(budget_manager=budget_manager, random_state=None)
        self.periodic_budget_manager = FixedBudget(budget=budget_manager.budget)
    
    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)
        
        _, utilities = self.budget_manager.sample(np.ones(len(X_cand)), simulate=simulate, return_budget_left=True)
        utilities = utilities.astype(np.float)
        sampled_indices = self.budget_manager.sample(utilities, simulate=simulate)
        
        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices