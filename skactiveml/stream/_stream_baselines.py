import numpy as np

from ..base import SingleAnnotatorStreamQueryStrategy
from ..utils import check_scalar


class StreamRandomSampling(SingleAnnotatorStreamQueryStrategy):
    """Random Sampling for Datastreams.

    The RandomSampling samples instances completely randomly. The
    probability to sample an instance is dependent on the budget specified in
    the budget manager. Given a budget of 10%, the utility exceeds 0.9 (1-0.1)
    with a probability of 10%. Instances are queried regardless of their
    position in the feature space. As this query strategy disregards any
    information about the instance. Thus, it should only be used as a baseline
    strategy.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    allow_exceeding_budget : bool, default=True
        If True, the query strategy is allowed to exceed it's budget as long as
        the average number of queries will be within the budget. If False,
        queries are not allowed if the budget is exhausted.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
            self, budget=None, allow_exceeding_budget=True, random_state=None
    ):
        super().__init__(budget=budget, random_state=random_state)
        self.allow_exceeding_budget = allow_exceeding_budget

    def query(self, candidates, return_utilities=False):
        """Ask the query strategy which instances in candidates to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        candidates : array-like or sparse matrix of shape
        (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in candidates which should be queried,
            with 0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        candidates, return_utilities = self._validate_data(
            candidates, return_utilities
        )

        # copy random state in case of simulating the query
        prior_random_state = self.random_state_.get_state()

        utilities = self.random_state_.random_sample(len(candidates))

        self.random_state_.set_state(prior_random_state)

        # keep record if the instance is queried and if there was budget left,
        # when assessing the corresponding utilities
        queried = np.full(len(utilities), False)

        # keep the internal state to reset it later if simulate is true
        tmp_observed_instances = self.observed_instances_
        tmp_queried_instances = self.queried_instances_
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, utility in enumerate(utilities):
            tmp_observed_instances += 1
            available_budget = (
                    tmp_observed_instances * self.budget_ - tmp_queried_instances
            )
            queried[i] = (
                                 self.allow_exceeding_budget or available_budget > 1
                         ) and (utility >= 1 - self.budget_)
            tmp_queried_instances += queried[i]

        # get the indices instances that should be queried
        queried_indices = np.where(queried)[0]

        # queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        candidates : array-like or sparse matrix of shape
        (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budgetmanager.

        Returns
        -------
        self : StreamRandomSampling
            The RandomSampling returns itself, after it is updated.
        """
        # check if a random state is set
        self._validate_data([[0]], False)
        # update observed instances and queried instances
        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self.observed_instances_ += candidates.shape[0]
        self.queried_instances_ += np.sum(queried)
        # update the random state assuming, that query(..., simulate=True) was
        # used
        self.random_state_.random_sample(len(candidates))
        return self

    def _validate_data(
            self,
            candidates,
            return_utilities,
            reset=True,
            **check_candidates_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        candidates: array-like of shape (n_candidates, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0

        check_scalar(
            self.allow_exceeding_budget, "allow_exceeding_budget", bool
        )

        candidates, return_utilities = super()._validate_data(
            candidates,
            return_utilities,
            reset=reset,
            **check_candidates_params
        )

        self._validate_random_state()

        return candidates, return_utilities


class PeriodicSampling(SingleAnnotatorStreamQueryStrategy):
    """The PeriodicSampling samples instances periodically. The length of that
    period is determined by the budget specified in the budgetmanager. For
    instance, a budget of 25% would result in the PeriodicSampling sampling
    every fourth instance. The main idea behind this query strategy is to
    exhaust a given budget as soon it is available. Instances are queried
    regardless of their position in the feature space. As this query strategy
    disregards any information about the instance. Thus, it should only be used
    as a baseline strategy.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget=None, random_state=None):
        super().__init__(budget=budget, random_state=random_state)

    def query(self, candidates, return_utilities=False):
        """Ask the query strategy which instances in candidates to acquire.

        This query strategy only evaluates the time each instance arrives at.
        The utilities returned, when return_utilities is set to True, are
        either 0 (the instance is not queried) or 1 (the instance is queried).
        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        candidates : array-like or sparse matrix of shape
        (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in candidates which should be queried,
            with 0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        candidates, return_utilities = self._validate_data(
            candidates, return_utilities
        )

        utilities = np.zeros(candidates.shape[0])

        # keep record if the instance is queried and if there was budget left,
        # when assessing the corresponding utilities
        queried = np.full(len(candidates), False)

        tmp_observed_instances = self.observed_instances_
        tmp_queried_instances = self.queried_instances_
        for i, x in enumerate(candidates):
            tmp_observed_instances += 1
            remaining_budget = (
                    tmp_observed_instances * self.budget_ - tmp_queried_instances
            )
            queried[i] = remaining_budget >= 1
            if queried[i]:
                utilities[i] = 1
            tmp_queried_instances += queried[i]

        # get the indices instances that should be queried
        queried_indices = np.where(queried)[0]

        # queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        candidates : array-like or sparse matrix of shape
        (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budgetmanager.

        Returns
        -------
        self : PeriodicSampling
            The PeriodicSampler returns itself, after it is updated.
        """
        # check if a budgetmanager is set
        self._validate_data(np.array([[0]]), False)
        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self.observed_instances_ += len(queried)
        self.queried_instances_ += np.sum(queried)
        return self

    def _validate_data(
            self,
            candidates,
            return_utilities,
            reset=True,
            **check_candidates_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        candidates: array-like of shape (n_candidates, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        candidates, return_utilities = super()._validate_data(
            candidates,
            return_utilities,
            reset=reset,
            **check_candidates_params
        )

        self._validate_random_state()

        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0

        return candidates, return_utilities
