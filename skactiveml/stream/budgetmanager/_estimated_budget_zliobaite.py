import numpy as np

from ...base import BudgetManager
from ...utils import check_scalar
from skactiveml.utils import check_classes
from skactiveml.utils._target import _check_class_vocabulary_structure
from skactiveml.utils._validation import (
    _ensure_random_state,
    _validate_budget_update,
)


class EstimatedBudgetZliobaite(BudgetManager):
    """EstimatedBudgetZliobaite

    Budget manager which checks, whether the specified `budget` has been
    exhausted already. If not, its subclasses decide whether a sample is
    queried.

    This budget manager calculates the estimated budget [1]_ spent in the last
    `w` steps and compares that to the `budget`. If the ratio is smaller than
    the specified budget, i.e., `budget - u_t / w > 0`, budget is left for
    querying a sample. `u` is the estimate of how many true labels were
    queried within the last `w` steps.
    The incremental function, `u_t = u_t-1 * (w-1) / w + labeling_t`, is used
    to calculate `u` at time `t`.

    Parameters
    ----------
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 < budget <= 1`. If `budget` is `None`, it is replaced with the
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
        queried_indices : array-like of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_candidates`.

        Returns
        -------
        self : EstimatedBudgetZliobaite
            The `EstimatedBudgetZliobaite` returns itself, after it is updated.
        """
        queried = _validate_budget_update(candidates, queried_indices)
        self._validate_data(np.array([]))
        for s in queried:
            self.u_t_ = self.u_t_ * ((self.w - 1) / self.w) + s

        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities : np.ndarray of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)

        if not hasattr(self, "u_t_"):
            self.u_t_ = 0

        return utilities


class FixedUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget Manager for Fixed Uncertainty Strategy

    Budget manager which implements the budgeting for the Fixed Uncertainty
    Strategy [1]_. If the `budget` is not exhausted, a sample is queried,
    when the probability of its most likely class, i.e., one minus its
    utility, does not exceed a threshold calculated based on the `budget` and
    the number of `classes`. See also :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    classes : array-like of shape (n_classes)
        Holds the label for each class.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 < budget <= 1`. If `budget` is `None`, it is replaced with the
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
        utilities : np.ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : list of int
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_samples`.
        """
        utilities = self._validate_data(utilities)
        confidence = 1 - utilities

        queried_indices = []
        budget_left = []
        theta = 1 / len(self.classes) + self.budget_ * (
            1 - 1 / len(self.classes)
        )

        tmp_u_t = self.u_t_

        samples = np.array(confidence) <= theta
        for i, d in enumerate(samples):
            budget_left.append(tmp_u_t / self.w < self.budget_)
            if not budget_left[-1]:
                d = False
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + d
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
        queried_indices : array-like of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_candidates`.

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
        utilities : np.ndarray of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)

        check_classes(self.classes)
        # The threshold below divides by the number of classes, so a nested
        # vocabulary would make it the number of label outputs.
        _check_class_vocabulary_structure("single-output", self.classes)
        return utilities


class VariableUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget Manager for Variable Uncertainty Strategy

    Budget manager which implements the budgeting for the Variable Uncertainty
    Strategy [1]_. Budget manager which checks, whether the specified budget
    has been exhausted already. If not, a sample is queried, when the
    probability of its most likely class, i.e., one minus its utility, is
    lower than `theta_`, which is a time-dependent threshold that decreases or
    increases when samples are queried or not queried, respectively. The rate
    for that change is controlled via `s`. See also
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    theta : float, default=1.0
        Specifies the initial value for `theta_`, the threshold for the
        probability of the most likely class.
    s : float, default=0.01
        Specifies the relative decrease or increase of the threshold if a
        sample is queried or not, respectively.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 < budget <= 1`. If `budget` is `None`, it is replaced with the
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
        utilities : np.ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : list of int
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_samples`.
        """
        utilities = self._validate_data(utilities)
        confidence = 1 - utilities

        queried_indices = []
        budget_left = []
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_

        for i, c in enumerate(confidence):
            budget_left.append(self.budget_ > tmp_u_t / self.w)

            if not budget_left[-1]:
                sample = False
            else:
                sample = c < tmp_theta
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
        queried_indices : array-like of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_candidates`.

        Returns
        -------
        self : VariableUncertaintyBudgetManager
            The budget manager returns itself, after it is updated.
        """
        queried = _validate_budget_update(candidates, queried_indices)
        self._validate_data(np.array([]))
        for s in queried:
            if self.budget_ > self.u_t_ / self.w:
                if s:
                    self.theta_ *= 1 - self.s
                else:
                    self.theta_ *= 1 + self.s
            # u_t = u_t-1 * (w-1)/w + labeling_t
            self.u_t_ = self.u_t_ * ((self.w - 1) / self.w) + s
        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities : np.ndarray of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        self._validate_theta()
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        check_scalar(
            self.s, "s", float, min_val=0, min_inclusive=False, max_val=1
        )

        return utilities

    def _validate_theta(self):
        """Validate if theta is set as a float."""
        check_scalar(self.theta, "theta", float)
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta


class RandomVariableUncertaintyBudgetManager(EstimatedBudgetZliobaite):
    """Budget Manager for Uncertainty Strategy With Randomization

    Budget manager which implements the budgeting for Uncertainty Strategy With
    Randomization [1]_. Budget manager which checks, whether the specified
    budget has been exhausted already. If not, a sample is queried, when the
    probability of its most likely class, i.e., one minus its utility, is
    lower than a randomized time-dependent threshold. The threshold is
    randomized by multiplying `theta_` with a random variable following a
    normal distribution with mean 1 and standard deviation `delta`. Similarly
    to :class:`.VariableUncertaintyBudgetManager`, `theta_` decreases or
    increases when samples are queried or not queried, respectively. The rate
    for that change is controlled via `s`. See also
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------

    delta : float, default=1.0
        Specifies the standard deviation of the normal distribution used for
        randomization of the threshold.
    theta : float, default=1.0
        Specifies the initial value for `theta_` that is used for calculating
        the threshold.
    s : float, default=0.01
        Specifies the relative decrease or increase of the threshold if a
        sample is queried or not, respectively.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the budget manager.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 < budget <= 1`. If `budget` is `None`, it is replaced with the
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
        utilities : np.ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : list of int
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_samples`.
        """
        utilities = self._validate_data(utilities)
        confidence = 1 - utilities

        queried_indices = []
        budget_left = []
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_

        prior_random_state = self.random_state_.get_state()

        for i, u in enumerate(confidence):
            budget_left.append(self.budget_ > tmp_u_t / self.w)

            if not budget_left[-1]:
                sample = False
            else:
                eta = self.random_state_.normal(1, self.delta)
                theta_random = tmp_theta * eta
                sample = u < theta_random
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
        queried_indices : array-like of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_candidates`.

        Returns
        -------
        self : RandomVariableUncertaintyBudgetManager
            The budget manager returns itself, after it is updated.
        """
        queried = _validate_budget_update(candidates, queried_indices)
        self._validate_data(np.array([]))
        for s in queried:
            if self.budget_ > self.u_t_ / self.w:
                # Replay the draw of `query_by_utility` for this sample.
                self.random_state_.normal(1, self.delta)
                if s:
                    self.theta_ *= 1 - self.s
                else:
                    self.theta_ *= 1 + self.s
            # u_t = u_t-1 * (w-1)/w + labeling_t
            self.u_t_ = self.u_t_ * ((self.w - 1) / self.w) + s
        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities : np.ndarray of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        self._validate_theta()
        check_scalar(
            self.s, "s", float, min_val=0, min_inclusive=False, max_val=1
        )
        check_scalar(
            self.delta, "delta", float, min_val=0, min_inclusive=False
        )
        _ensure_random_state(self)

        return utilities

    def _validate_theta(self):
        """Validate if theta is set as a float."""
        check_scalar(self.theta, "theta", float)
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta


class SplitBudgetManager(EstimatedBudgetZliobaite):
    """Budget Manager for Split Strategy

    Budget manager which implements the budgeting for the Split Strategy [1]_.
    The budget manager checks, whether the specified budget has been exhausted
    already. If not, a sample is queried, when the probability of its most
    likely class, i.e., one minus its utility, is lower than `theta_`, which
    is a time-dependent threshold that decreases or increases when samples are
    queried or not queried, respectively. The rate for that change is
    controlled via `s`. Additionally, with a probability of `v`, a sample is
    queried randomly with a probability of `budget` instead. See also
    :class:`.VariableUncertaintyBudgetManager` and
    :class:`.EstimatedBudgetZliobaite`.

    Parameters
    ----------
    v : float, default=0.1
        Probability of querying randomly with a probability of `budget`
        instead of comparing to `theta_`.
    theta : float, default=1.0
        Specifies the initial value for `theta_`, the threshold for the
        probability of the most likely class.
    s : float, default=0.01
        Specifies the relative decrease or increase of the threshold if a
        sample is queried or not, respectively.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the budget manager.
    w : int, default=100
        Specifies the size of the memory window. Controlls the `budget` in the
        last `w` steps taken.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 < budget <= 1`. If `budget` is `None`, it is replaced with the
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
        utilities : np.ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : list of int
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_samples`.
        """
        utilities = self._validate_data(utilities)
        confidence = 1 - utilities

        queried_indices = []
        budget_left = []
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        random_state_state = self.random_state_.get_state()

        for i, u in enumerate(confidence):
            budget_left.append(tmp_u_t / self.w < self.budget_)
            if not budget_left[-1]:
                sample = False
            else:
                random_val = self.random_state_.random_sample()
                if self.v > random_val:
                    new_u = self.random_state_.random_sample()
                    sample = new_u <= self.budget_ and not np.isnan(u)
                else:
                    sample = u < tmp_theta
                    if sample:
                        tmp_theta *= 1 - self.s
                    else:
                        tmp_theta *= 1 + self.s
                if sample:
                    queried_indices.append(i)

            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + sample

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
        queried_indices : array-like of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_candidates`.

        Returns
        -------
        self : SplitBudgetManager
            The budget manager returns itself, after it is updated.
        """
        queried = _validate_budget_update(candidates, queried_indices)
        self._validate_data(np.array([]))
        for q in queried:
            if self.u_t_ / self.w < self.budget_:
                if self.v > self.random_state_.random_sample():
                    _ = self.random_state_.random_sample()
                else:
                    if q:
                        self.theta_ *= 1 - self.s
                    else:
                        self.theta_ *= 1 + self.s
            # u_t = u_t-1 * (w-1)/w + labeling_t
            self.u_t_ = self.u_t_ * ((self.w - 1) / self.w) + q
        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities : np.ndarray of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        self._validate_theta()
        check_scalar(
            self.s, "s", float, min_val=0, min_inclusive=False, max_val=1
        )
        check_scalar(
            self.v,
            "v",
            float,
            min_val=0,
            min_inclusive=False,
            max_inclusive=False,
            max_val=1,
        )
        _ensure_random_state(self)

        return utilities

    def _validate_theta(self):
        """Validate if theta is set as a float."""
        check_scalar(self.theta, "theta", float)
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta


class RandomBudgetManager(EstimatedBudgetZliobaite):
    """RandomBudgetManager

    Budget manager which checks, whether the specified budget has been
    exhausted already. If not, samples are queried randomly with a
    probability of `budget`. See also
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
        `0 < budget <= 1`. If `budget` is `None`, it is replaced with the
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
        utilities : np.ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : list of int
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_samples`.
        """
        utilities = self._validate_data(utilities)
        confidence = 1 - utilities

        queried_indices = []

        tmp_u_t = self.u_t_

        prior_random_state = self.random_state_.get_state()

        samples = (
            self.random_state_.random_sample(len(confidence)) <= self.budget_
        )
        for i, d in enumerate(samples):
            budget_left = tmp_u_t / self.w < self.budget_
            d = d if budget_left else False
            tmp_u_t = tmp_u_t * ((self.w - 1) / self.w) + (
                d and not np.isnan(utilities[i])
            )
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
        queried_indices : array-like of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices < n_candidates`.

        Returns
        -------
        self : RandomBudgetManager
            The budget manager returns itself, after it is updated.
        """
        super().update(candidates, queried_indices)
        self.random_state_.random_sample(len(candidates))
        return self

    def _validate_data(self, utilities):
        """Validate input data.

        Parameters
        ----------
        utilities : np.ndarray of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities : ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        utilities = super()._validate_data(utilities)
        check_scalar(self.w, "w", int, min_val=0, min_inclusive=False)
        _ensure_random_state(self)

        return utilities
