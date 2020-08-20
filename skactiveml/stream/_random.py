import numpy as np

from sklearn.utils import check_array

from ..base import StreamBasedQueryStrategy


class RandomSampler(StreamBasedQueryStrategy):
    def __init__(self, budget_manager, random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)

        utilities = self.random_state.random_sample(len(X_cand))

        sampled_indices = self.budget_manager.sample(utilities,
                                                     simulate=simulate)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices


class PeriodicSampler(StreamBasedQueryStrategy):
    def __init__(self, budget_manager, random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.seen_instances = 0
        self.sampled_instances = 0

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)

        instances_to_sample = 0
        utilities = np.zeros(X_cand.shape[0])
        for i, x in enumerate(X_cand):
            seen_instances = (self.seen_instances + i + 1)
            sampled_instances = (self.sampled_instances + instances_to_sample)
            remaining_budget = (seen_instances * self.budget_manager.budget
                                - sampled_instances)
            if remaining_budget >= 1:
                utilities[i] = 1
                instances_to_sample += 1
            else:
                utilities[i] = 0

        if not simulate:
            self.seen_instances += X_cand.shape[0]
            self.sampled_instances += instances_to_sample
        sampled_indices = self.budget_manager.sample(utilities,
                                                     simulate=simulate)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices
