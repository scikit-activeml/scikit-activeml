import unittest
import numpy as np

from sklearn.datasets import make_classification

from ...classifier import PWC
from .._random import PeriodicSampler, RandomSampler


class TestRandom(unittest.TestCase):
    def setUp(self):
        # initialise valid data to test uncertainty parameters
        rand = np.random.RandomState(0)
        stream_length = 1000
        train_init_size = 10
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )

        self.X = X[:train_init_size, :]
        self.X_cand = X[train_init_size:, :]
        self.y = y[:train_init_size]
        self.clf = PWC()
        self.kwargs = dict(
            X_cand=self.X_cand, clf=self.clf, X=self.X, y=self.y
        )

    def test_periodic_sampler(self):
        # init param test
        self._test_init_param_budget_manager(PeriodicSampler)

        # update test
        self._test_update_without_query(PeriodicSampler)

    def test_random_sampler(self):
        # init param test
        self._test_init_param_budget_manager(RandomSampler)

        # update test
        self._test_update_without_query(RandomSampler)

    def _test_init_param_budget_manager(self, query_strategy_name):
        # budget_manager must be defined as an object of an budget manager
        # class
        query_strategy = query_strategy_name(budget_manager=[])
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))

    def _test_update_without_query(self, query_strategy_name):
        qs = query_strategy_name()
        qs.update(np.array([[0], [1], [2]]), np.array([0, 2]))
