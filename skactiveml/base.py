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

    def __init__(self,budget_manager, random_state=None):
        super().__init__(random_state=random_state)
        self.budget_manager = budget_manager

    @abstractmethod
    def query(self, X_cand, *args, return_utilities=False, simulate=False, **kwargs):
        return NotImplemented