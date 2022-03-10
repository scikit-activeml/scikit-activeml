from copy import deepcopy

import numpy as np

from ...base import BudgetManager
from ...utils import check_random_state, check_scalar


class EstimatedBudgetZliobaite(BudgetManager):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is queried, when the utility is
    higher than the specified budget.

    This budget manager calculates the estimated budget spent in the last
    w steps and compares that to the budget. If the ratio is smaller
    than the specified budget, i.e., budget - u_t / w > 0, the budget
    manager samples an instance when its utility is higher than the budget.
    u is the estimate of how many true lables were queried within the last
    w steps. The incremental funktion, u_t = u_t-1 * (w-1) / w + labeling_t,
    is used to calculate u at time t.

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1.
    w : int
        Specifies the size of the memory window. Controlles the budget in the
        last w steps taken. Default = 100
    """

    def __init__(self, budget=None, w=100):
        super().__init__(budget)
        self.w = w

    def update(self, candidates, queried_indices):
        """Updates the budget manager.

        Parameters
        ----------
        queried : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : EstimatedBudgetZliobaite
            The EstimatedBudgetZliobaite returns itself, after it is updated.
        """
        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self._validate_data(np.array([]))
        # update u_t for queried candidates
        for s in queried:
            self.u_t_ = self.u_t_ * ((self.w - 1) / self.w) + s

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

        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_t_"):
            self.u_t_ = 0

        return utilities


class FixedUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget manager which is optimized for FixedUncertainty and checks,
    whether the specified budget has been exhausted already. If not, an
    instance is queried, when the utility is higher than the specified budget
    and the probability of the most likely class exceeds a threshold
    calculated based on the budget and the number of classes.
    See also :class:`.EstimatedBudgetZliobaite`

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1.
    w : int
        Specifies the size of the memory window. Controlles the budget in the
        last w steps taken. Default = 100
    num_classes : int
        Specifies the number of classes. Default = 2
    """

    def __init__(self, budget=None, w=100, num_classes=2):
        super().__init__(budget, w)
        self.num_classes = num_classes

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
            queried, with 0 <= n_queried_instances <= n_samples.
        """
        utilities = self._validate_data(utilities)

        # intialize return parameters
        queried_indices = []
        budget_left = []
        # calculate theta with num_classes
        theta = 1 / self.num_classes + self.budget_ * (
            1 - 1 / self.num_classes
        )

        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_

        samples = np.array(utilities) <= theta
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, d in enumerate(samples):
            budget_left.append(tmp_u_t / self.w < self.budget_)
            if not budget_left[-1]:
                d = False
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + d
            # get the indices instances that should be queried
            if d:
                queried_indices.append(i)

            return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager.

        Parameters
        ----------
        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : FixedUncertaintyBudgetManager
            The FixedUncertaintyBudget returns itself, after it is updated.
        """
        super().update(candidates, queried_indices)
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
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        check_scalar(
            self.num_classes,
            "num_classes",
            int,
            min_val=0,
            min_inclusive=False,
        )

        return utilities


class VariableUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is queried, when the utility is
    higher than the specified budget and when the probability of
    the most likely class exceeds a time-dependent threshold calculated based
    on the budget, the number of classes and the number of observed and
    acquired samples.

    This budget manager calculates the estimated budget spent in the last
    w steps and compares that to the budget. If the ratio is smaller
    than the specified budget, i.e.,
    budget - u_t / w > 0 , the budget
    manager samples an instance when its utility is higher than the budget.
    u is the estimate of how many true lables were queried within the last
    w steps. The recursive funktion,
    u_t = u_t-1 * (w-1) / w + labeling_t , is used to calculate u at time t.
    See also :class:`.EstimatedBudgetZliobaite`

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1.
    w : int
        Specifies the size of the memory window. Controlles the budget in the
        last w steps taken. Default = 100
    theta : float
        Specifies the starting threshold in wich instances are purchased. This
        value of theta will recalculated after each instance. Default = 1
    s : float
        Specifies the value in wich theta is decresed or increased based on the
        purchase of the given label. Default = 0.01
    """

    def __init__(self, budget=None, w=100, theta=1.0, s=0.01):
        super().__init__(budget, w)
        self.theta = theta
        self.s = s

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
            queried, with 0 <= n_queried_instances <= n_samples.
        """
        utilities = self._validate_data(utilities)

        # intialize return parameters
        queried_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_

        # get utilities
        for i, u in enumerate(utilities):
            budget_left.append(self.budget_ > tmp_u_t / self.w)

            if not budget_left[-1]:
                sample = False
            else:
                sample = u < tmp_theta
                # get the indices instances that should be queried
                if sample:
                    tmp_theta *= 1 - self.s
                    queried_indices.append(i)
                else:
                    tmp_theta *= 1 + self.s
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

        return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager.

        Parameters
        ----------
        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : VariableUncertaintyBudgetManager
            The VariableUncertaintyBudget returns itself, after it is updated.
        """
        self._validate_data(np.array([]))

        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        for i, s in enumerate(queried):
            if self.budget_ > self.u_t_ / self.w:
                if s:
                    self.theta_ *= 1 - self.s
                else:
                    self.theta_ *= 1 + self.s
        super().update(candidates, queried_indices)
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
        # Check w
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        # Check theta
        self._validate_theta()
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        # Chack s
        check_scalar(
            self.s, "s", float, min_val=0, min_inclusive=False, max_val=1
        )

        return utilities

    def _validate_theta(self):
        """Validate if theta is set as a float."""
        check_scalar(self.theta, "theta", float)
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta


class RandomVariableUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is queried, when the utility is
    higher than the specified budget and when the probability of
    the most likely class exceeds a time-dependent threshold calculated based
    on the budget, the number of classes and the number of observed and
    acquired samples.

    This budget manager calculates the estimated budget spent in the last
    w steps and compares that to the budget. If the ratio is smaller
    than the specified budget, i.e.,
    budget - u_t / w > 0 , the budget
    manager samples an instance when its utility is higher than the budget.
    u is the estimate of how many true lables were queried within the last
    w steps. The recursive funktion,
    u_t = u_t-1 * (w-1) / w + labeling_t , is used to calculate u at time t.
    See also :class:`.EstimatedBudgetZliobaite`

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1.
    w : int
        Specifies the size of the memory window. Controlles the budget in the
        last w steps taken. Default = 100
    theta : float
        Specifies the starting threshold in wich instances are purchased. This
        value of theta will recalculated after each instance. Default = 1
    s : float
        Specifies the value in wich theta is decresed or increased based on the
        purchase of the given label. Default = 0.01
    """

    def __init__(
        self,
        budget=None,
        w=100,
        theta=1.0,
        s=0.01,
        delta=1.0,
        random_state=None,
    ):
        super().__init__(budget, w)
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
        return_utilities : bool, optional
            If true, also return whether there was budget left for each
            assessed utility. The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances represented by utilities which should be
            queried, with 0 <= n_queried_instances <= n_samples.
        """
        utilities = self._validate_data(utilities)

        # intialize return parameters
        queried_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_

        prior_random_state = self.random_state_.get_state()

        # get utilities
        for i, u in enumerate(utilities):
            budget_left.append(self.budget_ > tmp_u_t / self.w)

            if not budget_left[-1]:
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
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

        self.random_state_.set_state(prior_random_state)

        return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager.

        Parameters
        ----------
        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : RandomVariableUncertaintyBudgetManager
            The RandomVariableUncertaintyBudget returns itself, after it is
            updated.
        """
        self._validate_data(np.array([]))

        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self.random_state_.random_sample(len(candidates))
        for s in queried:
            if self.budget_ > self.u_t_ / self.w:
                if s:
                    self.theta_ *= 1 - self.s
                else:
                    self.theta_ *= 1 + self.s
        super().update(candidates, queried_indices)
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
        # Check w
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
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


class SplitBudgetManager(EstimatedBudgetZliobaite):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is queried, when the utility is
    higher than the specified budget. 100*v% of instances will be queried
    randomly and in 100*(1-v)% of will be queried cases according
    to VariableUncertainty

    This budget manager calculates the estimated budget spent in the last
    w steps and compares that to the budget. If the ratio is smaller
    than the specified budget, i.e., budget - u_t / w > 0 , the budget
    manager samples an instance when its utility is higher than the budget.
    u is the estimate of how many true lables were queried within the last
    w steps. The recursive funktion,
    u_t = u_t-1 * (w-1) / w + labeling_t , is used to calculate u at time t.
    See also :class:`.EstimatedBudgetZliobaite`

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be queried, with
        0 <= budget <= 1.
    w : int
        Specifies the size of the memory window. Controlles the budget in the
        last w steps taken. Default = 100
    theta : float
        Specifies the starting threshold in wich instances are purchased. This
        value of theta will recalculated after each instance. Default = 1
    s : float
        Specifies the value in wich theta is decresed or increased based on the
        purchase of the given label. Default = 0.01
    v : float
        Specifies the percent value of instances queried randomly.
    """

    def __init__(
        self, budget=None, w=100, theta=1.0, s=0.01, v=0.1, random_state=0
    ):
        super().__init__(budget, w)
        self.v = v
        self.theta = theta
        self.s = s
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

        # intialise return parameters
        queried_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        random_state_state = self.random_state_.get_state()

        # check for each queried separately if budget is left and the utility
        # is high enough
        for i, u in enumerate(utilities):
            budget_left.append(tmp_u_t / self.w < self.budget_)
            if not budget_left[-1]:
                sample = False
            else:
                # changed self.v < self.rand_.random_sample()
                random_val = self.random_state_.random_sample()
                if self.v > random_val:
                    new_u = self.random_state_.random_sample()
                    sample = new_u <= self.budget_
                else:
                    sample = u < tmp_theta
                    # get the indices instances that should be queried
                    if sample:
                        tmp_theta *= 1 - self.s
                    else:
                        tmp_theta *= 1 + self.s
                if sample:
                    queried_indices.append(i)

            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

        # set the internal state to the previous value
        self.random_state_.set_state(random_state_state)

        return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager.

        Parameters
        ----------
        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : SplitBudgetManager
            The SplitBudget returns itself, after it is updated.
        """
        self._validate_data(np.array([]))

        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        for x_t, q in zip(candidates, queried):
            if self.u_t_ / self.w < self.budget_:
                if self.v > self.random_state_.random_sample():
                    _ = self.random_state_.random_sample()
                else:
                    if q:
                        self.theta_ *= 1 - self.s
                    else:
                        self.theta_ *= 1 + self.s
            new_queried_indices = [0] if q else []
            super().update([x_t], new_queried_indices)
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
        # Check w
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        # Check theta
        self._validate_theta()
        # Check s
        check_scalar(
            self.s, "s", float, min_val=0, min_inclusive=False, max_val=1
        )
        # Check v
        check_scalar(
            self.v, "v", float, min_val=0, min_inclusive=False, max_val=1
        )
        # Check random_state
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
