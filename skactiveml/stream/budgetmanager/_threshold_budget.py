import numpy as np
from copy import deepcopy

from skactiveml.base import (
    BudgetManager,
)

from skactiveml.utils import check_scalar, check_random_state


class DensityBasedSplitBudgetManager(BudgetManager):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is queried, when the utility is
    higher than the specified budget and when the probability of
    the most likely class exceeds a time-dependent threshold calculated based
    on the budget, the number of classes and the number of observed and
    acquired samples. This class`s logic is the same as compared to
    SplitBudgetManager except for how available budget is calculated.

    This budget manager calculates the fixed budget spent and compares that to
    the budget. If the ratio is smaller
    than the specified budget, i.e., budget - u / t > 0 , the budget
    manager samples an instance when its utility is higher than the budget.
    u is the number of queried instances within t observed instances.

    Parameters
    ----------
    budget : float, optional (default=None)
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1. See Also :class:`BudgetManager`.
    theta : float, optional (default=1.0)
        Specifies the starting threshold in wich instances are purchased. This
        value of theta will recalculated after each instance. Default = 1
    s : float, optional (default=0.01)
        Specifies the value in wich theta is decresed or increased based on the
        purchase of the given label. Default = 0.01
    delta : float, optional (default=1.0)
        Specifies the standart deviation of the distribution. Default 1.0
    random_state : int | np.random.RandomState, optional (default=None)
        Random state for candidate selection.

    See Also
    --------
    EstimatedBudgetZliobaite : BudgetManager implementing the base class for
        Zliobaite based budget managers
    SplitBudgetManager : BudgetManager that is using EstimatedBudgetZliobaite.
    """

    def __init__(
        self,
        budget=None,
        theta=1.0,
        s=0.01,
        delta=1.0,
        random_state=None,
    ):
        super().__init__(budget)
        self.theta = theta
        self.s = s
        self.delta = delta
        self.random_state = random_state

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
        confidence = 1 - utilities

        # intialize return parameters
        queried_indices = []
        tmp_u = self.u_
        tmp_t = self.t_
        tmp_theta = self.theta_

        prior_random_state = self.random_state_.get_state()

        # get confidence
        for i, u in enumerate(confidence):
            tmp_t += 1
            budget_left = self.budget_ > tmp_u / tmp_t
            if not budget_left:
                sample = False
            else:
                eta = self.random_state_.normal(1, self.delta)
                theta_random = tmp_theta * eta
                sample = u < theta_random
                # get the indices instances that should be queried
                if sample:
                    tmp_theta *= 1 - self.s
                    queried_indices.append(i)
                else:
                    tmp_theta *= 1 + self.s
            tmp_u += sample

        self.random_state_.set_state(prior_random_state)

        return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : DensityBasedBudgetManager
            The DensityBasedBudgetManager returns itself, after it is
            updated.
        """
        self._validate_data(np.array([]))

        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self.random_state_.random_sample(len(candidates))
        for s in queried:
            self.t_ += 1
            if self.budget_ > self.u_ / self.t_:
                if s:
                    self.theta_ *= 1 - self.s
                else:
                    self.theta_ *= 1 + self.s
            self.u_ += s

        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities: ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy.


        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked utilities.
        """

        utilities = super()._validate_data(utilities)
        # Check theta
        self._validate_theta()
        # Chack s
        check_scalar(
            self.s, "s", float, min_val=0, min_inclusive=False, max_val=1
        )
        # Check delta
        check_scalar(
            self.delta, "delta", float, min_val=0, min_inclusive=False
        )
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_"):
            self.u_ = 0
        if not hasattr(self, "t_"):
            self.t_ = 0
        self._validate_random_state()

        return utilities

    def _validate_theta(self):
        """Validate if theta is set as a float."""
        check_scalar(self.theta, "theta", float)
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = deepcopy(self.random_state)
        self.random_state_ = check_random_state(self.random_state_)
