import numpy as np

from ..base import SingleAnnotatorStreamQueryStrategy
from ..utils import check_scalar


class StreamRandomSampling(SingleAnnotatorStreamQueryStrategy):
    """Random Sampling for Data Streams.

    The RandomSampling strategy queries labels completely randomly. The
    probability to query a sample is dependent on the budget specified in
    the budget manager. Given a budget of 10%, the utility exceeds 0.9 (1-0.1)
    with a probability of 10%. samples are queried regardless of their
    position in the feature space and disregards any information about the
    sample. Thus, it should only be used as a baseline strategy. The
    `allow_exceeding_budget` parameter allows to configure the strategy to
    strictly adhere to a given budget.

    Parameters
    ----------
    allow_exceeding_budget : bool, default=True
        If `True`, the query strategy is allowed to exceed it's budget as long
        as the average number of queries will be within the budget. If `False`,
        queries are not allowed if the budget is exhausted.
    budget : float, default=None
        The budget which models the budgeting constraint used in the
        stream-based active learning setting.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self, allow_exceeding_budget=True, budget=None, random_state=None
    ):
        super().__init__(budget=budget, random_state=random_state)
        self.allow_exceeding_budget = allow_exceeding_budget

    def query(self, candidates, return_utilities=False):
        """Determines for which candidate samples labels are to be queried.

        The query startegy determines the most useful samples in candidates,
        which can be acquired within the budgeting constraint specified by
        `budget`. Please note that, this method does not change the internal
        state of the query strategy. To adapt the query strategy to the
        selected candidates, use `update(...)`.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool, default=False
            If `True`, also return the `utilities` based on the query strategy.

        Returns
        -------
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        utilities: np.ndarray of shape (n_candidates,),
            The utilities based on the query strategy. Only provided if
            `return_utilities` is `True`.
        """
        candidates, return_utilities = self._validate_data(
            candidates, return_utilities
        )

        # copy random state in case of simulating the query
        prior_random_state = self.random_state_.get_state()

        utilities = self.random_state_.random_sample(len(candidates))

        self.random_state_.set_state(prior_random_state)

        # keep record if the sample is queried and if there was budget left,
        # when assessing the corresponding utilities
        queried = np.full(len(utilities), False)

        # keep the internal state to reset it later if simulate is true
        tmp_observed_samples = self.observed_samples_
        tmp_queried_samples = self.queried_samples_
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, utility in enumerate(utilities):
            tmp_observed_samples += 1
            available_budget = (
                tmp_observed_samples * self.budget_ - tmp_queried_samples
            )
            queried[i] = (
                self.allow_exceeding_budget or available_budget > 1
            ) and (utility >= 1 - self.budget_)
            tmp_queried_samples += queried[i]

        # get the indices samples that should be queried
        queried_indices = np.where(queried)[0]

        # queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the count for seen and queried labels. This function should
        be used in conjunction with the `query` function.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.

        Returns
        -------
        self : SingleAnnotatorStreamQueryStrategy
            The query strategy returns itself, after it is updated.
        """
        # check if a random state is set
        self._validate_data([[0]], False)
        # update observed samples and queried samples
        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self.observed_samples_ += candidates.shape[0]
        self.queried_samples_ += np.sum(queried)
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
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute. If False, the
            input will be checked for consistency with data provided when reset
            was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        # check if counting of samples has begun
        if not hasattr(self, "observed_samples_"):
            self.observed_samples_ = 0
        if not hasattr(self, "queried_samples_"):
            self.queried_samples_ = 0

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
    """Periodic Sampling for Data Streams

    The PeriodicSampling strategy samples labels periodically. The length of
    that period is determined by the `budget`. For instance, a `budget` of 0.25
    would result in querying every fourth sample. The main idea behind this
    query strategy is to exhaust a given budget as soon as it is available.
    samples are queried regardless of their position in the feature space and
    disregards any information about the sample. Thus, it should only be used
    as a baseline strategy.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in the
        stream-based active learning setting.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget=None, random_state=None):
        super().__init__(budget=budget, random_state=random_state)

    def query(self, candidates, return_utilities=False):
        """Determines for which candidate samples labels are to be queried.

        The query startegy determines the most useful samples in candidates,
        which can be acquired within the budgeting constraint specified by
        `budget`. Please note that, this method does not change the internal
        state of the query strategy. To adapt the query strategy to the
        selected candidates, use `update(...)`.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool, default=False
            If `True`, also return the `utilities` based on the query strategy.

        Returns
        -------
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        utilities: np.ndarray of shape (n_candidates,),
            The utilities based on the query strategy. Only provided if
            `return_utilities` is `True`.
        """
        candidates, return_utilities = self._validate_data(
            candidates, return_utilities
        )

        utilities = np.zeros(candidates.shape[0])

        # keep record if the sample is queried and if there was budget left,
        # when assessing the corresponding utilities
        queried = np.full(len(candidates), False)

        tmp_observed_samples = self.observed_samples_
        tmp_queried_samples = self.queried_samples_
        for i, x in enumerate(candidates):
            tmp_observed_samples += 1
            remaining_budget = (
                tmp_observed_samples * self.budget_ - tmp_queried_samples
            )
            queried[i] = remaining_budget >= 1
            if queried[i]:
                utilities[i] = 1
            tmp_queried_samples += queried[i]

        # get the indices samples that should be queried
        queried_indices = np.where(queried)[0]

        # queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, candidates, queried_indices):
        """Updates the count for seen and queried labels. This function should
        be used in conjunction with the `query` function.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted only
            if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried, with
            `0 <= queried_indices <= n_candidates`.

        Returns
        -------
        self : SingleAnnotatorStreamQueryStrategy
            The query strategy returns itself, after it is updated.
        """
        # check if a budgetmanager is set
        self._validate_data(np.array([[0]]), False)
        queried = np.zeros(len(candidates))
        queried[queried_indices] = 1
        self.observed_samples_ += len(queried)
        self.queried_samples_ += np.sum(queried)
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
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute. If False, the
            input will be checked for consistency with data provided when reset
            was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples.
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

        # check if counting of samples has begun
        if not hasattr(self, "observed_samples_"):
            self.observed_samples_ = 0
        if not hasattr(self, "queried_samples_"):
            self.queried_samples_ = 0

        return candidates, return_utilities
