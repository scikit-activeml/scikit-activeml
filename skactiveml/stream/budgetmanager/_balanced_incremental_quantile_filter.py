from collections import deque
from copy import copy

import numpy as np

from ...base import BudgetManager
from ...utils import check_scalar


class BalancedIncrementalQuantileFilter(BudgetManager):
    """Balanced Incremental Quantile Filter (BIQF)

    The Balanced Incremental Quantile Filter has been proposed together with
    Probabilistic Active Learning for Datastreams [1]_. It assesses whether a
    given spatial utility (i.e., obtained via :class:`ProbabilisticAL`)
    warrants to query the label in question. The spatial ultilities are
    compared against a threshold that is derived from a quantile (`budget`) of
    the last `w` observed utilities. To balance the number of queries, `w_tol`
    is used to increase or decrease the threshold based on the number of
    available acquisitions.

    Parameters
    ----------
    w : int, default=100
        The number of observed utilities that are used to infer the threshold.
        `w` should be higher than 0.
    w_tol : int, default=50
        The window in which the number of acquisitions should stay within the
        budget. `w_tol` should be higher than 0.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] D. Kottke, G. Krempl, and M. Spiliopoulou. Probabilistic Active
        Learning in Datastreams. In Adv. Intell. Data Anal., pages 145–157,
        2015.
    """

    def __init__(self, w=100, w_tol=50, budget=None):
        super().__init__(budget)
        self.w = w
        self.w_tol = w_tol

    def query_by_utility(self, utilities):
        """Ask the budget manager which `utilities` are sufficient to query the
        corresponding labels.

        Parameters
        ----------
        utilities : array-like of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        """
        utilities = self._validate_data(utilities)

        # intialize return parameters
        queried_indices = []

        tmp_queried_samples_ = self.queried_samples_
        tmp_observed_samples_ = self.observed_samples_
        tmp_history_sorted_ = copy(self.history_sorted_)

        for i, u in enumerate(utilities):
            tmp_observed_samples_ += 1
            tmp_history_sorted_.append(u)
            theta = np.quantile(tmp_history_sorted_, (1 - self.budget_))

            min_ranking = np.min(tmp_history_sorted_)
            max_ranking = np.max(tmp_history_sorted_)
            range_ranking = max_ranking - min_ranking

            acq_left = (
                self.budget_ * tmp_observed_samples_ - tmp_queried_samples_
            )
            theta_bal = theta - (range_ranking * (acq_left / self.w_tol))
            sample = u >= theta_bal

            if sample:
                tmp_queried_samples_ += 1
                queried_indices.append(i)

        return queried_indices

    def update(self, candidates, queried_indices, utilities):
        """Updates the budget manager.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_samples, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.

        Returns
        -------
        self : BalancedIncrementalQuantileFilter
            The budget manager returns itself, after it is updated.
        """
        self._validate_data(np.array([0]))

        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1

        self.observed_samples_ += len(queried)
        self.queried_samples_ += np.sum(queried)
        self.history_sorted_.extend(utilities)

        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        check_scalar(
            self.w_tol, "w_tol", (float, int), min_val=0, min_inclusive=False
        )

        # check if counting of samples has begun
        if not hasattr(self, "observed_samples_"):
            self.observed_samples_ = 0
        if not hasattr(self, "queried_samples_"):
            self.queried_samples_ = 0
        if not hasattr(self, "history_sorted_"):
            self.history_sorted_ = deque(maxlen=self.w)

        return utilities
