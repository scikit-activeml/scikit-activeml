import numpy as np

from ..base import SingleAnnotStreamBasedQueryStrategy

from .budget_manager import FixedThresholdBudget
from ..utils import call_func


class RandomSampler(SingleAnnotStreamBasedQueryStrategy):
    """The RandomSampler samples instances completely randomly. The probability
    to sample an instance is dependent on the budget specified in the
    budget_manager. Given a budget of 10%, the utility exceeds 0.9 (1-0.1) with
    a probability of 10%. Instances are queried regardless of their position in
    the feature space. As this query strategy disregards any information about
    the instance. Thus, it should only be used as a baseline strategy.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedThresholdBudget will be used by default.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget_manager=None,
                 random_state=None):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(self, X_cand, return_utilities=False):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        X_Cand, return_utilities = self._validate_data(
            X_cand, return_utilities
        )

        # copy random state in case of simulating the query
        prior_random_state_state = self.random_state_.get_state()

        utilities = self.random_state_.random_sample(len(X_cand))

        self.random_state_.set_state(prior_random_state_state)

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, X_cand, queried_indices, budget_manager_param_dict=None):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : RandomSampler
            The RandomSampler returns itself, after it is updated.
        """
        # check if a random state is set
        self._validate_random_state()
        # check if a budget_manager is set
        self._validate_budget_manager()
        budget_manager_param_dict = ({} if budget_manager_param_dict is None
                                     else budget_manager_param_dict)
        # update the random state assuming, that query(..., simulate=True) was
        # used
        self.random_state_.random_sample(len(X_cand))
        call_func(
            self.budget_manager_.update,
            X_cand=X_cand,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return FixedThresholdBudget()

    def _validate_data(
        self, X_cand, return_utilities, reset=True, **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like of shape (n_candidates, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        X_cand: np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        X_cand, return_utilities = super()._validate_data(
            X_cand, return_utilities, reset=reset, **check_X_cand_params
        )

        self._validate_random_state()

        return X_cand, return_utilities


class PeriodicSampler(SingleAnnotStreamBasedQueryStrategy):
    """The PeriodicSampler samples instances periodically. The length of that
    period is determined by the budget specified in the budget_manager. For
    instance, a budget of 25% would result in the PeriodicSampler sampling
    every fourth instance. The main idea behind this query strategy is to
    exhaust a given budget as soon it is available. Instances are queried
    regardless of their position in the feature space. As this query strategy
    disregards any information about the instance. Thus, it should only be used
    as a baseline strategy.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedThresholdBudget will be used by default.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget_manager=None,
                 random_state=None):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(self, X_cand, return_utilities=False):
        """Ask the query strategy which instances in X_cand to acquire.

        This query strategy only evaluates the time each instance arrives at.
        The utilities returned, when return_utilities is set to True, are
        either 0 (the instance is not queried) or 1 (the instance is queried).
        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        X_cand, return_utilities = self._validate_data(
            X_cand,
            return_utilities
            )

        utilities = np.zeros(X_cand.shape[0])
        budget = getattr(self.budget_manager_, "budget_", 0)

        tmp_observed_instances = self.observed_instances_
        tmp_queried_instances = self.queried_instances_
        for i, x in enumerate(X_cand):
            tmp_observed_instances += 1
            remaining_budget = (
                tmp_observed_instances * budget - tmp_queried_instances
            )
            if remaining_budget >= 1:
                utilities[i] = 1
                tmp_queried_instances += 1
            else:
                utilities[i] = 0

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, X_cand, queried_indices, budget_manager_param_dict=None):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : PeriodicSampler
            The PeriodicSampler returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_data(np.array([[0]]), False)
        budget_manager_param_dict = ({} if budget_manager_param_dict is None
                                     else budget_manager_param_dict)
        call_func(
            self.budget_manager_.update,
            X_cand=X_cand,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        queried = np.zeros(len(X_cand))
        queried[queried_indices] = 1
        self.observed_instances_ += len(queried)
        self.queried_instances_ += np.sum(queried)
        # print("queried_instances_", self.queried_instances_)
        return self

    def get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return FixedThresholdBudget()

    def _validate_data(
        self, X_cand, return_utilities, reset=True, **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like of shape (n_candidates, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        X_cand: np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        X_cand, return_utilities = super()._validate_data(
            X_cand, return_utilities, reset=reset, **check_X_cand_params
        )

        self._validate_random_state()

        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0

        return X_cand, return_utilities
