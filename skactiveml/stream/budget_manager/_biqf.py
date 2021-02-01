import numpy as np

from .base import BudgetManager, get_default_budget
# from sortedcontainers import SortedList
from collections import deque


class BIQF(BudgetManager):
    """
    """
    def __init__(self, w, w_tol, budget=None):
        super().__init__(budget)
        self.w = w
        self.w_tol = w_tol

    def sample(
        self, utilities, return_budget_left=False, simulate=False, **kwargs
    ):
        # check if budget has been set
        self._validate_budget(get_default_budget())
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        if not hasattr(self, "theta_"):
            self.theta_ = 0.0
        if not hasattr(self, "theta_bal_"):
            self.theta_bal_ = 0.0
        if not hasattr(self, "history_sorted"):
            self.history_sorted = deque(maxlen=self.w)
        
        # intialize return parameters
        sampled_indices = []

        for i, u in enumerate(utilities):
            self.observed_instances_ += 1
            self.history_sorted.append(u)
            self.theta_ = np.quantile(self.history_sorted, (1 - self.budget_))

            range_ranking = np.max(self.historyArr) - (
                np.min(self.historyArr)) + 1e-6
            acq_left = self.budget * self.observed_instances_ - (
                self.queried_instances_)
            self.theta_bal = self.theta_ - range_ranking * acq_left / (
                self.w_tol)
            
            sample = u >= self.theta_bal
            
            if sample:
                self.queried_instances_ += 1
            sampled_indices.append(sample)

        return sampled_indices
