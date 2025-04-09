from copy import deepcopy

import numpy as np

from ...base import BudgetManager
from ...utils import check_random_state, check_scalar
from skactiveml.utils import check_classes


class EstimatedBudgetZliobaite(BudgetManager):
    """EstimatedBudgetZliobaite

    Budget manager which checks, whether the specified `budget` has been
    exhausted already. If not, a sample is queried, when the utility is
    higher than the specified `budget`.

    This budget manager calculates the estimated budget [1]_ spent in the last
    `w` steps and compares that to the `budget`. If the ratio is smaller than
    the specified budget, i.e., `budget - u_t / w > 0`, the budget manager
    queries a sample when its utility is higher than the budget. `u` is the
    estimate of how many true labels were queried within the last `w` steps.
    The incremental function, `u_t = u_t-1 * (w-1) / w + labeling_t`, is used
    to calculate `u` at time `t`.

    Parameters
    ----------
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def __init__(self, w=100, budget=None):
        super().__init__(budget)
        self.w = w

    def update(self, candidates, queried_indices):
        """Updates the `EstimatedBudgetZliobaite`.

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
        self : EstimatedBudgetZliobaite
            The `EstimatedBudgetZliobaite` returns itself, after it is updated.
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
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)

        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, "u_t_"):
            self.u_t_ = 0

        return utilities


class FixedUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget Manager for Fixed Uncertainty Strategy

    Budget manager which implements the budgeting for the Fixed Uncertainty
    Strategy [1]_. If the not `budget` is not exhausted, a sample is
    queried, when the utility is higher than the specified `budget` and the
    probability of the most likely class exceeds a threshold calculated based
    on the `budget` and the number of `classes`. See also
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    classes : array-like of shape (n_classes)
        Holds the label for each class.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def __init__(self, classes, w=100, budget=None):
        super().__init__(w=w, budget=budget)
        self.classes = classes

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
        confidence = 1 - utilities

        # intialize return parameters
        queried_indices = []
        budget_left = []
        # calculate theta with number of classes
        theta = 1 / len(self.classes) + self.budget_ * (
            1 - 1 / len(self.classes)
        )

        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_

        samples = np.array(confidence) <= theta
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, d in enumerate(samples):
            budget_left.append(tmp_u_t / self.w < self.budget_)
            if not budget_left[-1]:
                d = False
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + d
            # get the indices samples that should be queried
            if d:
                queried_indices.append(i)

        return queried_indices

    def update(self, candidates, queried_indices):
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
        self : FixedUncertaintyBudgetManager
            The budget manager returns itself, after it is updated.
        """
        super().update(candidates, queried_indices)
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

        check_classes(self.classes)
        return utilities


class VariableUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget Manager for Variable Uncertainty Strategy

    Budget manager which implements the budgeting for the Variable Uncertainty
    Strategy [1]_. Budget manager which checks, whether the specified budget
    has been exhausted already. If not, a sample is queried, when the
    utility is higher than `theta_`, which is a time-dependent threshold that
    increases or decreases when samples are queried or not queried,
    respectively. The rate for that change is controlled via `s`. See also
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    theta : float, default=1.0
        Specifies the initial value for `theta_` that is compared to
        `utilities`.
    s : float, default=0.1
        Specifies the relative increase or decrease of the threshold if an
        sample is queried or not, respectively.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def __init__(self, theta=1.0, s=0.01, w=100, budget=None):
        super().__init__(w=w, budget=budget)
        self.theta = theta
        self.s = s

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
        confidence = 1 - utilities

        # intialize return parameters
        queried_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_

        # get confidence
        for i, c in enumerate(confidence):
            budget_left.append(self.budget_ > tmp_u_t / self.w)

            if not budget_left[-1]:
                sample = False
            else:
                sample = c < tmp_theta
                # get the indices samples that should be queried
                if sample:
                    tmp_theta *= 1 - self.s
                    queried_indices.append(i)
                else:
                    tmp_theta *= 1 + self.s
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

        return queried_indices

    def update(self, candidates, queried_indices):
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
        self : VariableUncertaintyBudgetManager
            The budget manager returns itself, after it is updated.
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
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
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
    """Budget Manager for Uncertainty Strategy With Randomization

    Budget manager which implements the budgeting for Uncertainty Strategy With
    Randomization [1]_. Budget manager which checks, whether the specified
    budget has been exhausted already. If not, a sample is queried, when the
    utility is higher than a randomized time-dependent threshold. The threshold
    is rendomized by multiplying `theta_` with a random variable following a
    normal distribution with mean 1 and standard deviation `mu`. Similarly, to
    :class:`.VariableUncertaintyBudgetManager`, `theta_` increases or decreases
    when samples are queried or not queried, respectively. The rate for that
    change is controlled via `s`. See also :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------

    delta : float, default=1.0
        Specifies the standart deviation of the normal distribution used for
        randomization of the threshold.
    theta : float, default=1.0
        Specifies the initial value for `theta_` that is used for calculating
        the threshold.
    s : float, default=0.1
        Specifies the relative increase or decrease of the threshold if an
        sample is queried or not, respectively.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the budget manager.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def __init__(
        self,
        delta=1.0,
        theta=1.0,
        s=0.01,
        random_state=None,
        w=100,
        budget=None,
    ):
        super().__init__(w=w, budget=budget)
        self.delta = delta
        self.theta = theta
        self.s = s
        self.random_state = random_state

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
        confidence = 1 - utilities

        # intialize return parameters
        queried_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_

        prior_random_state = self.random_state_.get_state()

        # get confidence
        for i, u in enumerate(confidence):
            budget_left.append(self.budget_ > tmp_u_t / self.w)

            if not budget_left[-1]:
                sample = False
            else:
                eta = self.random_state_.normal(1, self.delta)
                theta_random = tmp_theta * eta
                sample = u < theta_random
                # get the indices samples that should be queried
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
        candidates : {array-like, sparse matrix} of shape\
                (n_samples, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.

        Returns
        -------
        self : RandomVariableUncertaintyBudgetManager
            The budget manager returns itself, after it is updated.
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
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
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
    """Budget Manager for Split Strategy

    Budget manager which implements the budgeting for the Split Strategy [1]_.
    The budget manager checks, whether the specified budget has been exhausted
    already. If not, a sample is queried, when the utility is higher than
    `theta_`, which is a time-dependent threshold that increases or decreases
    when samples are queried or not queried, respectively. The rate for that
    change is controlled via `s`. Additionally, samples are queried randomly
    with a probability of `v`. See also
    :class:`.VariableUncertaintyBudgetManager` and
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    v : float, default=0.1
        Specifies the percent value of samples queried randomly.
    theta : float, default=1.0
        Specifies the initial value for `theta_` that is compared to
        `utilities`.
    s : float, default=0.1
        Specifies the relative increase or decrease of the threshold if an
        sample is queried or not, respectively.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the budget manager.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def __init__(
        self, v=0.1, theta=1.0, s=0.01, random_state=None, w=100, budget=None
    ):
        super().__init__(w=w, budget=budget)
        self.v = v
        self.theta = theta
        self.s = s
        self.random_state = random_state

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
        confidence = 1 - utilities

        # intialise return parameters
        queried_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        random_state_state = self.random_state_.get_state()

        # check for each queried separately if budget is left and the utility
        # is high enough
        for i, u in enumerate(confidence):
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
                    # get the indices samples that should be queried
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
        candidates : {array-like, sparse matrix} of shape\
                (n_samples, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.

        Returns
        -------
        self : SplitBudgetManager
            The budget manager returns itself, after it is updated.
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
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
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
            self.v,
            "v",
            float,
            min_val=0,
            min_inclusive=False,
            max_inclusive=False,
            max_val=1,
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


class RandomBudgetManager(EstimatedBudgetZliobaite):
    """RandomBudgetManager

    Budget manager which checks, whether the specified budget has been
    exhausted already. If not, a sample is queried, when the utility is
    higher than the specified budget. If budget is available, samples are
    queried randomly with a probability of `budget` %. See also
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the budget manager.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    """

    def __init__(self, random_state=None, w=100, budget=None):
        super().__init__(w=w, budget=budget)
        self.random_state = random_state

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
        confidence = 1 - utilities

        # intialize return parameters
        queried_indices = []

        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_

        prior_random_state = self.random_state_.get_state()

        samples = (
            self.random_state_.random_sample(len(confidence)) <= self.budget_
        )
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, d in enumerate(samples):
            budget_left = tmp_u_t / self.w < self.budget_
            d = d if budget_left else False
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + (
                d and not np.isnan(utilities[i])
            )
            # get the indices samples that should be queried
            if d and not np.isnan(utilities[i]):
                queried_indices.append(i)

        self.random_state_.set_state(prior_random_state)

        return queried_indices

    def update(self, candidates, queried_indices):
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
        self : SplitBudgetManager
            The budget manager returns itself, after it is updated.
        """
        self._validate_data(np.array([]))
        self.random_state_.random_sample(len(candidates))
        super().update(candidates, queried_indices)
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
        self._validate_random_state()

        return utilities

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = deepcopy(self.random_state)
        self.random_state_ = check_random_state(self.random_state_)
