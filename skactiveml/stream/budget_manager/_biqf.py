import numpy as np

from skactiveml.base import BudgetManager
from collections import deque
from copy import copy


class BIQF(BudgetManager):
    """
    """

    def __init__(self, w=100, w_tol=50, budget=None, save_utilities=True):
        super().__init__(budget)
        self.w = w
        self.w_tol = w_tol
        self.save_utilities = save_utilities

    def is_budget_left(self):
        return True

    def sample(
        self, utilities, return_budget_left=False, simulate=False, **kwargs
    ):
        """Ask the budget manager which utilities are sufficient to sample the
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

        simulate : bool, optional
            If True, the internal state of the budget manager before and after
            the query is the same. This should only be used to prevent the
            budget manager from adapting itself. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances represented by utilities which should be
            sampled, with 0 <= n_sampled_instances <= n_samples.

        budget_left: ndarray of shape (n_samples,), optional
            Shows whether there was budget left for each assessed utility. Only
            provided if return_utilities is True.
        """
        utilities, return_budget_left, simulate = self.validate_data(
            utilities, return_budget_left, simulate
        )

        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        if not hasattr(self, "history_sorted_"):
            self.history_sorted_ = deque(maxlen=self.w)
        if not hasattr(self, "utility_queue_"):
            self.utility_queue_ = deque(maxlen=self.w)
        # intialize return parameters
        sampled_indices = []

        tmp_queried_instances_ = self.queried_instances_
        tmp_observed_instances_ = self.observed_instances_
        tmp_history_sorted_ = copy(self.history_sorted_)
        tmp_utility_queue_ = copy(self.utility_queue_)

        for i, u in enumerate(utilities):
            tmp_observed_instances_ += 1
            tmp_history_sorted_.append(u)
            tmp_utility_queue_.append(u)
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
                sampled_indices.append(i)

        if not simulate:
            self.queried_instances_ = tmp_queried_instances_
            self.observed_instances_ = tmp_observed_instances_
            self.history_sorted_ = tmp_history_sorted_
        else:
            self.utility_queue_ = tmp_utility_queue_

        return sampled_indices

    def update(self, sampled, utilities=None, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        utilities : ndarray of shape (n_samples,), optional
            The utilities based on the query strategy.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """
        # check if budget has been set
        self._validate_budget()
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        if not hasattr(self, "history_sorted_"):
            self.history_sorted_ = deque(maxlen=self.w)
        self.observed_instances_ += len(sampled)
        self.queried_instances_ += np.sum(sampled)
        if utilities is not None:
            self.history_sorted_.extend(utilities)
        else:
            if self.save_utilities:
                utilities = [self.utility_queue_.popleft() for _ in sampled]
                self.history_sorted_.extend(utilities)
            else:
                raise ValueError(
                    "The save_utilities variable has to be set to true, when"
                    + " no utilities are passed to update"
                )

        return self

    def validate_data(self, utilities, return_budget_left, simulate):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            candidate samples
        return_budget_left : bool,
            If true, also return the budget based on the query strategy.
        simulate : bool,
            If True, the internal state of the budget manager before and after
            the query is the same.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked candidate samples
        return_budget_left : bool,
            Checked boolean value of `return_budget_left`.
        simulate : bool,
            Checked boolean value of `simulate`.
        """

        utilities, return_budget_left, simulate = super()._validate_data(
            utilities, return_budget_left, simulate
        )
        self._validate_w()
        self._validate_w_tol()
        self._validate_save_utilities()

        return utilities, return_budget_left, simulate

    def _validate_w_tol(self):
        """Validate if w_tol is set as an int and greater than 0.
        """
        if not (isinstance(self.w_tol, int) or isinstance(self.w_tol, float)):
            raise TypeError(
                "{} is not a valid type for w_tol".format(type(self.w_tol))
            )
        if self.w_tol <= 0:
            raise ValueError(
                "The value of w_tol is incorrect."
                + " w_tol must be greater than 0"
            )

    def _validate_w(self):
        """Validate if w is set as an int and greater than 0.
        """
        if not isinstance(self.w, int):
            raise TypeError(
                "{} is not a valid type for w".format(type(self.w))
            )
        if self.w <= 0:
            raise ValueError(
                "The value of w is incorrect." + " w must be greater than 0"
            )

    def _validate_save_utilities(self):
        """Validate if save_utilities is set as a bool and initialize
        self.utility_queue_ accordingly.
        """
        if isinstance(self.save_utilities, bool):
            if self.save_utilities and not hasattr(self, "utility_queue_"):
                self.utility_queue_ = deque(maxlen=self.w)
        else:
            raise TypeError("save_utilities is not a boolean.")

