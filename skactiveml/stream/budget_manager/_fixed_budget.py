import numpy as np

from skactiveml.base import BudgetManager


class FixedBudget(BudgetManager):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is sampled, when the utility is
    higher than the specified budget.
    This budget manager counts the number of already observed instances and
    compares that to the number of sampled instances. If the ratio is smaller
    than the specified budget, i.e.,
    n_observed_instances * budget - n_sampled_instances >= 1 , the budget
    manager samples an instance when its utility is higher than the budget.
    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1.
    """

    def __init__(self, budget=None):
        super().__init__(budget)

    def query(
        self, utilities, **kwargs
    ):
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
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        # keep record if the instance is queried and if there was budget left,
        # when assessing the corresponding utilities
        queried = np.full(len(utilities), False)
        budget_left = np.full(len(utilities), False)

        # keep the internal state to reset it later if simulate is true
        tmp_observed_instances = self.observed_instances_
        tmp_queried_instances = self.queried_instances_
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, utility in enumerate(utilities):
            tmp_observed_instances += 1
            budget_left[i] = (
                tmp_observed_instances * self.budget_ - tmp_queried_instances
            )
            queried[i] = budget_left[i] and (utility >= 1 - self.budget_)
            tmp_queried_instances += queried[i]

        # get the indices instances that should be queried
        queried_indices = np.where(queried)[0]

        return queried_indices

    def update(self, X_cand, queried_indices, **kwargs):
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
        queried = np.zeros(len(X_cand))
        queried[queried_indices] = 1
        # check if budget has been set
        self._validate_budget()
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        self.observed_instances_ += X_cand.shape[0]
        self.queried_instances_ += np.sum(queried)
        return self