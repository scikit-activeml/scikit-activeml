import numpy as np
from copy import deepcopy

from skactiveml.base import BudgetManager

from skactiveml.utils import check_scalar, check_random_state


class DensityBasedSplitBudgetManager(BudgetManager):
    """Budget Manager for DBALStream

    This budget manager is an adaptation of
    :class:`.RandomVariableUncertaintyBudgetManager` for DBALStream [1]_. It
    mainly differs in how the available budget ist estimated. Instead of the
    estimated budget proposed by Žliobaitė et. al. [2]_, this budget manager
    counts the number of queried and seen instance, such that the number of
    available queries is given as `n_seen_samples-n_queried_samples*budget`.

    Parameters
    ----------
    theta : float, default=1.0
        Specifies the initial value for `theta_` that is used for calculating
        the threshold.
    s : float, default=0.1
        Specifies the relative increase or decrease of the threshold if an
        sample is queried or not, respectively.
    delta : float, default=1.0
        Specifies the standart deviation of the normal distribution used for
        randomization of the threshold.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the budget manager.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.

    References
    ----------
    .. [1] D. Ienco, I. Žliobaitė, and B. Pfahringer. High density-focused
        uncertainty sampling for active learning over evolving stream data. In
        Int. Workshop Big Data Streams Heterog. Source Min. Algorithms Syst.
        Program. Models Appl., pages 133–148, 2014.
    .. [2] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def __init__(
        self,
        theta=1.0,
        s=0.01,
        delta=1.0,
        random_state=None,
        budget=None,
    ):
        super().__init__(budget)
        self.theta = theta
        self.s = s
        self.delta = delta
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
                # get the indices samples that should be queried
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
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
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
