import numpy as np
from copy import deepcopy

from .base import BudgetManager, get_default_budget
from skactiveml.utils import check_random_state


class EstimatedBudget(BudgetManager):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is sampled, when the utility is
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
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1.
    w : int
        Specifies the size of the memory window. Controlles the budget in the
        last w steps taken. Default = 100
    """

    def __init__(self, budget=None, w=100):
        super().__init__(budget)
        self.w = w

    def is_budget_left(self):
        """Check whether there is any utility given to sample(...), which may
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
        return self.budget_ > self.u_t_ / self.w

    def update(self, sampled, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """
        # check if budget has been set
        self._validate_budget(get_default_budget())
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_t_"):
            self.u_t_ = 0
        # update u_t for sampled X_cand
        for s in sampled:
            self.u_t_ = self.u_t_ * ((self.w - 1) / self.w) + s

        return self


class FixedUncertaintyBudget(EstimatedBudget):
    """Budget manager which is optimized for FixedUncertainty and checks,
    whether the specified budget has been exhausted already. If not, an
    instance is sampled, when the utility is higher than the specified budget
    and the probability of the most likely class exceeds a threshold
    calculated based on the budget and the number of classes.
    See also :class:`.EstimatedBudget`

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
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
        utilities, return_budget_left, simulate = self._validate_data(
            utilities, return_budget_left, simulate
        )
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_t_"):
            self.u_t_ = 0

        # intialize return parameters
        sampled_indices = []
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
            # get the indices instances that should be sampled
            if d:
                sampled_indices.append(i)

        # set the internal state to the previous values
        if not simulate:
            self.u_t_ = tmp_u_t

        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices

    def update(self, sampled, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """

        super().update(sampled)
        return self

    def _validate_data(self, utilities, return_budget_left, simulate):
        """Validate input data.

        Parameters
        ----------
        utilities: ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy.
        return_budget_left : bool,
            If true, also return the budget based on the query strategy.
        simulate : bool,
            If True, the internal state of the budget manager before and after
            the query is the same.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked utilities.
        return_budget_left : bool,
            Checked boolean value of `return_budget_left`.
        simulate : bool,
            Checked boolean value of `simulate`.
        """

        utilities, return_budget_left, simulate = super()._validate_data(
            utilities, return_budget_left, simulate
        )
        self._validate_w()
        self._validate_num_classes()

        return utilities, return_budget_left, simulate

    def _validate_num_classes(self):
        """Validate if num_classes is an integer and greater than 0.
        """
        if not isinstance(self.num_classes, int):
            raise TypeError("{} is not a valid type for num_classes")
        if self.num_classes <= 0:
            raise ValueError(
                "The value of num_classes is incorrect."
                + " num_classes must be greater than 0"
            )

    def _validate_w(self):
        """Validate if w is an integer and greater than 0.
        """
        if not isinstance(self.w, int):
            raise TypeError("{} is not a valid type for w")
        if self.w <= 0:
            raise ValueError(
                "The value of w is incorrect." + " w must be greater than 0"
            )


class VarUncertaintyBudget(EstimatedBudget):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is sampled, when the utility is
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

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
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
        utilities, return_budget_left, simulate = self._validate_data(
            utilities, return_budget_left, simulate
        )
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_t_"):
            self.u_t_ = 0

        # intialize return parameters
        sampled_indices = []
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
                # get the indices instances that should be sampled
                if sample:
                    tmp_theta *= 1 - self.s
                    sampled_indices.append(i)
                else:
                    tmp_theta *= 1 + self.s
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

        # set the internal state to the previous values
        if not simulate:
            self.u_t_ = tmp_u_t
            self.theta_ = tmp_theta

        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices

    def update(self, sampled, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """

        for i, s in enumerate(sampled):
            if self.is_budget_left():
                if s:
                    self.theta_ *= 1 - self.s
                else:
                    self.theta_ *= 1 + self.s
            super().update([s])
        return self

    def _validate_data(self, utilities, return_budget_left, simulate):
        """Validate input data.

        Parameters
        ----------
        utilities: ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy.
        return_budget_left : bool,
            If true, also return the budget based on the query strategy.
        simulate : bool,
            If True, the internal state of the budget manager before and after
            the query is the same.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked utilities.
        return_budget_left : bool,
            Checked boolean value of `return_budget_left`.
        simulate : bool,
            Checked boolean value of `simulate`.
        """

        utilities, return_budget_left, simulate = super()._validate_data(
            utilities, return_budget_left, simulate
        )
        # Check w
        self._validate_w()
        # Check theta
        self._validate_theta()
        # Chack s
        self._validate_s()

        return utilities, return_budget_left, simulate

    def _validate_theta(self):
        """Validate if theta is set as a float.
        """
        if not isinstance(self.theta, float):
            raise TypeError("{} is not a valid type for theta")

    def _validate_s(self):
        """Validate if s a float and in range (0,1].
        """
        if self.s is not None:
            if not isinstance(self.s, float):
                raise TypeError("{} is not a valid type for s")
            if self.s <= 0 or self.s > 1.0:
                raise ValueError(
                    "The value of s is incorrect."
                    + " s must be defined in range (0,1]"
                )

    def _validate_w(self):
        """Validate if w is an integer and greater than 0.
        """
        if not isinstance(self.w, int):
            raise TypeError("{} is not a valid type for w")
        if self.w <= 0:
            raise ValueError(
                "The value of w is incorrect." + " w must be greater than 0"
            )


class SplitBudget(EstimatedBudget):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is sampled, when the utility is
    higher than the specified budget. 100*v% of instances will be sampled
    randomly and in 100*(1-v)% of will be sampled cases according
    to VarUncertainty

    This budget manager calculates the estimated budget spent in the last
    w steps and compares that to the budget. If the ratio is smaller
    than the specified budget, i.e.,
    budget - u_t / w > 0 , the budget
    manager samples an instance when its utility is higher than the budget.
    u is the estimate of how many true lables were queried within the last
    w steps. The recursive funktion,
    u_t = u_t-1 * (w-1) / w + labeling_t , is used to calculate u at time t.

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
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
        Specifies the percent value of instances sampled randomly.
    """

    def __init__(
        self, budget=None, w=100, theta=1.0, s=0.01, v=0.1, random_state=0
    ):
        super().__init__(budget, w)
        self.v = v
        self.theta = theta
        self.s = s
        self.random_state = random_state

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
        utilities, return_budget_left, simulate = self._validate_data(
            utilities, return_budget_left, simulate
        )
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_t_"):
            self.u_t_ = 0

        # intialise return parameters
        sampled_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        random_state_state = self.random_state_.get_state()

        # check for each sample separately if budget is left and the utility is
        # high enough
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
                    # get the indices instances that should be sampled
                    if sample:
                        tmp_theta *= 1 - self.s
                    else:
                        tmp_theta *= 1 + self.s
                if sample:
                    sampled_indices.append(i)

            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

        # set the internal state to the previous values
        if simulate:
            self.random_state_.set_state(random_state_state)
        else:
            self.u_t_ = tmp_u_t
            self.theta_ = tmp_theta

        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices

    def update(self, sampled, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """

        for s in sampled:
            if self.u_t_ / self.w < self.budget_:  # self.is_budget_left():
                if self.v > self.random_state_.random_sample():
                    _ = self.random_state_.random_sample()
                else:
                    if s:
                        self.theta_ *= 1 - self.s
                    else:
                        self.theta_ *= 1 + self.s
            super().update([s])
        return self

    def _validate_data(self, utilities, return_budget_left, simulate):
        """Validate input data.

        Parameters
        ----------
        utilities: ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy.
        return_budget_left : bool,
            If true, also return the budget based on the query strategy.
        simulate : bool,
            If True, the internal state of the budget manager before and after
            the query is the same.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked utilities.
        return_budget_left : bool,
            Checked boolean value of `return_budget_left`.
        simulate : bool,
            Checked boolean value of `simulate`.
        """

        utilities, return_budget_left, simulate = super()._validate_data(
            utilities, return_budget_left, simulate
        )
        # Check w
        self._validate_w()
        # Check theta
        self._validate_theta()
        # Check s
        self._validate_s()
        # Check v
        self._validate_v()
        # Check random_state
        self._validate_random_state()

        return utilities, return_budget_left, simulate

    def _validate_theta(self):
        """Validate if theta is set as a float.
        """
        if not isinstance(self.theta, float):
            raise TypeError("{} is not a valid type for theta")

    def _validate_s(self):
        """Validate if s a float and in range (0,1].
        """
        if self.s is not None:
            if not isinstance(self.s, float):
                raise TypeError("{} is not a valid type for s")
            if self.s <= 0 or self.s > 1.0:
                raise ValueError(
                    "The value of s is incorrect."
                    + " s must be defined in range (0,1]"
                )

    def _validate_v(self):
        """Validate if v is a float and in range (0,1].
        """
        if not isinstance(self.v, float):
            raise TypeError("{} is not a valid type for v")
        if self.v <= 0 or self.v >= 1:
            raise ValueError(
                "The value of v is incorrect."
                + " v must be defined in range (0,1)"
            )

    def _validate_w(self):
        """Validate if w an integer and greater than 0.
        """
        if not isinstance(self.w, int):
            raise TypeError("{} is not a valid type for w")
        if self.w <= 0:
            raise ValueError(
                "The value of w is incorrect." + " w must be greater than 0"
            )

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = deepcopy(self.random_state)
        self.random_state_ = check_random_state(self.random_state_)
