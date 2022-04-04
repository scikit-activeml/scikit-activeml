from collections import deque
from copy import copy

import numpy as np

from ...base import BudgetManager
from ...utils import check_scalar


class BalancedIncrementalQuantileFilter(BudgetManager):
    """
    The Balanced Incremental Quantile Filter has been proposed together with
    Probabilistic Active Learning for Datastreams [1]. It assesses whether a
    given spatial utility (i.e., obtained via ProbabilisticAL) warrants to
    query the label in question. The spatial ultilities are compared against
    a threshold that is derived from a quantile (budget) of the last w observed
    utilities.
    To balance the number of queries, w_tol is used to increase or decrease the
    threshold based on the number of available acquisitions.

    Parameters
    ----------
    w : int
        The number of observed utilities that are used to infer the threshold.
        w should be higher than 0.

    w_tol : int
        The window in which the number of acquisitions should stay within the
        budget. w_tol should be higher than 0.

    budget : float
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1.

    References
    ----------
    [1] Kottke D., Krempl G., Spiliopoulou M. (2015) Probabilistic Active
        Learning in Datastreams. In: Fromont E., De Bie T., van Leeuwen M.
        (eds) Advances in Intelligent Data Analysis XIV. IDA 2015. Lecture
        Notes in Computer Science, vol 9385. Springer, Cham.
    """

    def __init__(self, w=100, w_tol=50, budget=None):
        super().__init__(budget)
        self.w = w
        self.w_tol = w_tol

    def query_by_utility(self, utilities):
        """Ask the budget manager which utilities are sufficient to query the
        corresponding instance.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether sampling an instance
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances represented by utilities which should be
            queried, with 0 <= n_queried_instances <= n_samples.
        """
        utilities = self._validate_data(utilities)

        # intialize return parameters
        queried_indices = []

        tmp_queried_instances_ = self.queried_instances_
        tmp_observed_instances_ = self.observed_instances_
        tmp_history_sorted_ = copy(self.history_sorted_)

        for i, u in enumerate(utilities):
            tmp_observed_instances_ += 1
            tmp_history_sorted_.append(u)
            theta = np.quantile(tmp_history_sorted_, (1 - self.budget_))

            min_ranking = np.min(tmp_history_sorted_)
            max_ranking = np.max(tmp_history_sorted_)
            range_ranking = max_ranking - min_ranking

            acq_left = (
                self.budget_ * tmp_observed_instances_ - tmp_queried_instances_
            )
            theta_bal = theta - (range_ranking * (acq_left / self.w_tol))
            sample = u >= theta_bal

            if sample:
                tmp_queried_instances_ += 1
                queried_indices.append(i)

        return queried_indices

    def update(self, candidates, queried_indices, utilities):
        """Updates the budget manager.

        Parameters
        ----------
        queried_indices : array-like
            Indicates which instances from candidates have been queried.

        utilities : ndarray of shape (n_samples,), optional
            The utilities based on the query strategy.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """

        self._validate_data(np.array([0]))

        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1

        self.observed_instances_ += len(queried)
        self.queried_instances_ += np.sum(queried)
        self.history_sorted_.extend(utilities)

        return self

    def _validate_data(self, utilities):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            candidate samples

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked candidate samples
        """

        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        check_scalar(
            self.w_tol, "w_tol", (float, int), min_val=0, min_inclusive=False
        )

        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        if not hasattr(self, "history_sorted_"):
            self.history_sorted_ = deque(maxlen=self.w)

        return utilities
