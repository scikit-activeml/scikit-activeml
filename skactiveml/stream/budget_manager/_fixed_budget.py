import numpy as np

from .base import BudgetManager


class FixedBudget(BudgetManager):

    def __init__(self, budget):
        super().__init__(budget)
        self.seen_samples = 0
        self.sampled_samples = 0

    def is_budget_left(self):
        return self.seen_samples * self.budget - self.sampled_samples >= 1

    def sample(self, utilities, simulate=False,
               **kwargs):
        sampled = np.full(len(utilities), False)
        budget_left = np.full(len(utilities), False)

        init_seen_samples = self.seen_samples
        init_sampled_samples = self.sampled_samples

        for i, utility in enumerate(utilities):
            self.seen_samples += 1
            budget_left[i] = self.is_budget_left()
            sampled[i] = budget_left[i] and (utility >= 1 - self.budget)
            self.sampled_samples += sampled[i]

        if simulate:
            self.seen_samples = init_seen_samples
            self.sampled_samples = init_sampled_samples

        sampled_indices = np.where(sampled)[0]

        return sampled_indices
