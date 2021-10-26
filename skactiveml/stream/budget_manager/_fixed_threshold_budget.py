import numpy as np

from ...base import BudgetManager
from ...utils._validation import check_scalar


class FixedThresholdBudget(BudgetManager):
    """The FixedThresholdBudget uses the provided budget as a threshold for the
    utility. If the utility exceeds the threshold (1-budget), the label is
    queried.  The utilities are expected to be within 0 and 1.
    This budget manager is able to count the number of already observed
    instances and monitor the number of queried labels. The flag
    allow_exceeding_budget specifies whether the given budget can be exceeded.
    If allow_exceeding_budget is set to False, the budget manager keeps track
    the number of queries ans only allows further queries, if the budget is not
    exhausted already.
    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1.
    allow_exceeding_budget : bool
        Specifies if the budget manager is able to exceed the budget.
    """

    def __init__(self, budget=None, allow_exceeding_budget=True):
        super().__init__(budget)
        self.allow_exceeding_budget = allow_exceeding_budget

    def query_by_utility(self, utilities):
        """Ask the budget manager which utilities are sufficient to query the
        corresponding instance.
        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether sampling an instance
            is worth it given the budgeting constraint.
        return_utilities : bool, optional
            If true, also return whether there was budget left for each
            assessed utility. The default is False.
        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances represented by utilities which should be
            sampled, with 0 <= n_queried_instances <= n_samples.
        """
        utilities = self._validate_data(utilities)
        # keep record if the instance is queried and if there was budget left,
        # when assessing the corresponding utilities
        queried = np.full(len(utilities), False)

        # keep the internal state to reset it later if simulate is true
        tmp_observed_instances = self.observed_instances_
        tmp_queried_instances = self.queried_instances_
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, utility in enumerate(utilities):
            tmp_observed_instances += 1
            available_budget = (
                tmp_observed_instances
                * self.budget_
                - tmp_queried_instances
            )
            queried[i] = (
                (self.allow_exceeding_budget or available_budget > 1)
                and (utility >= 1 - self.budget_)
            )
            tmp_queried_instances += queried[i]

        # get the indices instances that should be queried
        queried_indices = np.where(queried)[0]

        return queried_indices

    def update(self, X_cand, queried_indices):
        """Updates the budget manager.
        Parameters
        ----------
        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.
        Returns
        -------
        self : FixedBudget
            The FixedBudget returns itself, after it is updated.
        """
        self._validate_data(np.array([]))
        queried = np.zeros(len(X_cand))
        queried[queried_indices] = 1
        self.observed_instances_ += X_cand.shape[0]
        self.queried_instances_ += np.sum(queried)
        return self

    def _validate_data(self, utilities):
        utilities = super()._validate_data(utilities)
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0

        check_scalar(
            self.allow_exceeding_budget, "allow_exceeding_budget", bool
        )
        return utilities
