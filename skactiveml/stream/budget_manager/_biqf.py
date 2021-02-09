import numpy as np

from .base import BudgetManager, get_default_budget
from collections import deque


class BIQF(BudgetManager):
    """
    """

    def __init__(self, w=100, w_tol=50, budget=None):
        super().__init__(budget)
        self.w = w
        self.w_tol = w_tol

    def is_budget_left(self):
        pass

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
        if not hasattr(self, "history_sorted_"):
            self.history_sorted_ = deque(maxlen=self.w)
        # check if w is set
        if not isinstance(self.w, int):
            raise TypeError("{} is not a valid type for w")
        if self.w <= 0:
            raise ValueError(
                "The value of w is incorrect." + " w must be greater than 0"
            )
        # check if w_tol is set
        if not (isinstance(self.w_tol, int) or isinstance(self.w_tol, float)):
            raise TypeError("{} is not a valid type for w_tol")
        if self.w_tol <= 0:
            raise ValueError(
                "The value of w_tol is incorrect." 
                + " w_tol must be greater than 0"
            )
        # check if utilities is set
        if not isinstance(utilities, np.ndarray):
            raise TypeError("{} is not a valid type for utilities")

        # intialize return parameters
        sampled_indices = []

        for i, u in enumerate(utilities):
            self.observed_instances_ += 1
            self.history_sorted_.append(u)
            theta = np.quantile(self.history_sorted_, (1 - self.budget_))

            min_ranking = np.min(self.history_sorted_)
            max_ranking = np.max(self.history_sorted_)
            range_ranking = max_ranking - min_ranking

            acq_left = (
                self.budget_ * self.observed_instances_
                - self.queried_instances_
            )
            theta_bal = theta - range_ranking * acq_left / self.w_tol

            sample = u >= theta_bal

            if sample:
                self.queried_instances_ += 1
            sampled_indices.append(sample)

        return sampled_indices

    def update(self, sampled, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """
        # check if budget has been set
        self._validate_budget(get_default_budget())
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        self.observed_instances_ += sampled.shape[0]
        self.queried_instances_ += np.sum(sampled)
        return self

        return self
