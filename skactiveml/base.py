from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


class QueryStrategy(ABC, BaseEstimator):

    def __init__(self, random_state=None):
        # set RS
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def query(self, *args, **kwargs):
        return NotImplemented


class PoolBasedQueryStrategy(QueryStrategy):

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    @abstractmethod
    def query(self, X_cand, *args, return_utilities=False, **kwargs):
        return NotImplemented


class StreamBasedQueryStrategy(QueryStrategy):
    """Base class for all stream-based active learning query strategies in
       scikit-activeml.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """
    def __init__(self, budget_manager, random_state=None):
        super().__init__(random_state=random_state)
        self.budget_manager = budget_manager

    @abstractmethod
    def query(self, X_cand, *args, return_utilities=False, simulate=False,
              **kwargs):
        """Ask the query strategy which instance in X_cand to acquire.

        The query startegy determines the most useful instances in X_cand,
        which can be acquired within the budgeting constraint specified by the
        budget_manager.

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
        return NotImplemented
