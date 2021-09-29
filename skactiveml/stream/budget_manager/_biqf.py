import numpy as np

from skactiveml.base import BudgetManager
from collections import deque
from copy import copy


class BIQF(BudgetManager):
    """
    The Balanced Incremental Quantile Filter has been proposed together with
    Probabilistic Active Learning for Datastreams [1]. It assesses whether a given
    spatial utility (i.e., obtained via McPAL) warrants to query the label in
    question. The spatial ultilities are compared against a threshold that is
    derived from a quantile (budget) of the last w observed utilities. To
    balance the number of queries, w_tol is used to increase or decrease the
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

    save_utilities : bool
        A flag that controls whether the utilities for previous queries should
        be saved within the object. This flag affects whether the spatial
        utilities should be provided when using update.

    References
    ----------
    [1] Kottke D., Krempl G., Spiliopoulou M. (2015) Probabilistic Active
        Learning in Datastreams. In: Fromont E., De Bie T., van Leeuwen M.
        (eds) Advances in Intelligent Data Analysis XIV. IDA 2015. Lecture
        Notes in Computer Science, vol 9385. Springer, Cham.
    """

    def __init__(self, w=100, w_tol=50, budget=None, save_utilities=True):
        super().__init__(budget)
        self.w = w
        self.w_tol = w_tol
        self.save_utilities = save_utilities

    def is_budget_left(self):
        """Check whether there is any utility given to query(...), which may
            lead to sampling the corresponding instance, i.e., check if sampling
            another instance is currently possible under the budgeting constraint.
            This function is useful to determine, whether a provided
            utility is not sufficient, or the budgeting constraint was simply
            exhausted. For this budget manager this function returns True, when
            budget > estimated_spending.

            Returns
            -------
            budget_left : bool
                True, if there is a utility which leads to sampling another
                instance.
        """
        return True

    def query(
        self, utilities, simulate=False, return_budget_left=False, **kwargs
    ):
        """Ask the budget manager which utilities are sufficient to query the
        corresponding instance.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether sampling an instance
            is worth it given the budgeting constraint.
        simulate : bool, optional
            If True, the internal state of the budget manager before and after
            the query is the same. This should only be used to prevent the
            budget manager from adapting itself. The default is False.
        return_utilities : bool, optional
            If true, also return whether there was budget left for each
            assessed utility. The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances represented by utilities which should be
            queried, with 0 <= n_queried_instances <= n_samples.

        budget_left: ndarray of shape (n_samples,), optional
            Shows whether there was budget left for each assessed utility. Only
            provided if return_utilities is True.
        """
        utilities, simulate, return_budget_left = self.validate_data(
            utilities, simulate, return_budget_left
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
        queried_indices = []

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
                queried_indices.append(i)

        if not simulate:
            self.queried_instances_ = tmp_queried_instances_
            self.observed_instances_ = tmp_observed_instances_
            self.history_sorted_ = tmp_history_sorted_
        else:
            self.utility_queue_ = tmp_utility_queue_

        return queried_indices

    def update(self, queried, utilities=None, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        queried : array-like
            Indicates which instances from X_cand have been queried.

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
        self.observed_instances_ += len(queried)
        self.queried_instances_ += np.sum(queried)
        if utilities is not None:
            self.history_sorted_.extend(utilities)
        else:
            if self.save_utilities:
                utilities = [self.utility_queue_.popleft() for _ in queried]
                self.history_sorted_.extend(utilities)
            else:
                raise ValueError(
                    "The save_utilities variable has to be set to true, when"
                    + " no utilities are passed to update"
                )

        return self

    def validate_data(self, utilities, simulate, return_budget_left):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            candidate samples
        simulate : bool,
            If True, the internal state of the budget manager before and after
            the query is the same.
        return_budget_left : bool,
            If true, also return the budget based on the query strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked candidate samples
        return_budget_left : bool,
            Checked boolean value of `return_budget_left`.
        simulate : bool,
            Checked boolean value of `simulate`.
        """

        utilities, simulate, return_budget_left = super()._validate_data(
            utilities, simulate, return_budget_left
        )
        self._validate_w()
        self._validate_w_tol()
        self._validate_save_utilities()

        return utilities, simulate, return_budget_left

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

