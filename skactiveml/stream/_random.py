import numpy as np

from sklearn.utils import check_array

from ..base import SingleAnnotStreamBasedQueryStrategy

from .budget_manager import FixedBudget


class RandomSampler(SingleAnnotStreamBasedQueryStrategy):
    """The RandomSampler samples instances completely randomly. The probability
    to sample an instance is dependent on the budget specified in the
    budget_manager. Given a budget of 10%, the utility exceeds 0.9 (1-0.1) with
    a probability of 10%. Instances are sampled regardless of their position in
    the feature space. As this query strategy disregards any information about
    the instance. Thus, it should only be used as a baseline strategy.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. The budget attribute set for
        the budget_manager will be used to determine the probability to sample
        instances

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget_manager=FixedBudget(), random_state=None):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        simulate : bool, optional
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Member variabled created
            during the query call may not be deleted, however, their state is
            reset in such a way, as the query call never happened. Note, that
            this parameter is propagated to the budget_manager, as well.
            The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a random state is set
        self._validate_random_state()
        # copy random state in case of simulating the query
        prior_random_state_state = self.random_state_.get_state()
        # check if a budget_manager is set
        self._validate_budget_manager()

        utilities = self.random_state_.random_sample(len(X_cand))

        sampled_indices = self.budget_manager_.sample(
            utilities, simulate=simulate
        )

        if simulate:
            self.random_state_.set_state(prior_random_state_state)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : RandomSampler
            The RandomSampler returns itself, after it is updated.
        """
        # check if a random state is set
        self._validate_random_state()
        # check if a budget_manager is set
        self._validate_budget_manager()
        # update the random state assuming, that query(..., simulate=True) was
        # used
        self.random_state_.random_sample(len(sampled))
        self.budget_manager_.update(sampled)
        return self


class PeriodicSampler(SingleAnnotStreamBasedQueryStrategy):
    """The PeriodicSampler samples instances periodically. The length of that
    period is determined by the budget specified in the budget_manager. For
    instance, a budget of 25% would result in the PeriodicSampler sampling
    every fourth instance. The main idea behind this query strategy is to
    exhaust a given budget as soon it is available. Instances are sampled
    regardless of their position in the feature space. As this query strategy
    disregards any information about the instance. Thus, it should only be used
    as a baseline strategy.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. The budget attribute set for
        the budget_manager will be used to determine the interval between
        sampling instnces

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget_manager=FixedBudget(), random_state=None):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
        """Ask the query strategy which instances in X_cand to acquire.

        This query strategy only evaluates the time each instance arrives at.
        The utilities returned, when return_utilities is set to True, are
        either 0 (the instance is not sampled) or 1 (the instance is sampled).
        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        simulate : bool, optional
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Note, that this is propagated
            to the budget_manager, as well. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0

        utilities = np.zeros(X_cand.shape[0])
        budget = getattr(self.budget_manager_, "budget_", 0)

        tmp_observed_instances = self.observed_instances_
        tmp_queried_instances = self.queried_instances_
        for i, x in enumerate(X_cand):
            tmp_observed_instances += 1
            remaining_budget = (
                tmp_observed_instances * budget - tmp_queried_instances
            )
            # print(remaining_budget >= 1)
            if remaining_budget >= 1:
                utilities[i] = 1
                tmp_queried_instances += 1
            else:
                utilities[i] = 0

            # print("observed_instances", observed_instances)
            # print("queried_instances", queried_instances)
            # print("budget", budget)
            # print("remaining_budget", remaining_budget)

        if not simulate:
            self.observed_instances_ = tmp_observed_instances
            self.queried_instances_ = tmp_queried_instances
        sampled_indices = self.budget_manager_.sample(
            utilities, simulate=simulate
        )
        # print("sampled_indices", sampled_indices)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : PeriodicSampler
            The PeriodicSampler returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0

        self.budget_manager_.update(sampled)
        self.observed_instances_ += X_cand.shape[0]
        self.queried_instances_ += np.sum(sampled > 0)
        # print("queried_instances_", self.queried_instances_)
        return self
