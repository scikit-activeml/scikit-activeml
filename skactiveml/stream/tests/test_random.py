import unittest
import numpy as np

from sklearn.datasets import make_classification

from skactiveml.stream import PeriodicSampler, RandomSampler


class TemplateTestRandom:
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

        self.X = X[[train_init_size], :]
        self.X_cand = X[train_init_size:, :]
        self.y = y[:train_init_size]
        self.kwargs = dict(
            X_cand=self.X_cand
        )

    def test_init_param_budget_manager(self):
        # budget_manager must be defined as an object of an budget manager
        # class
        query_strategy = self.get_query_strategy()(budget_manager=[])
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))

    def test_query_param_X_cand(self):
        # X_cand must be defined as a two dimensinal array
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=1
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=None
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=np.ones(5)
        )

    def test_init_param_random_state(self):
        query_strategy = self.get_query_strategy()(random_state="string",)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))

    def test_query_param_return_utilities(self):
        # return_utilities needs to be a boolean
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            return_utilities="string",
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            return_utilities=1,
        )

    def test_update_without_query(self):
        qs = self.get_query_strategy()()
        qs.update(np.array([[0], [1], [2]]), np.array([0, 2]))


class TestRandomSampler(TemplateTestRandom, unittest.TestCase):
    def get_query_strategy(self):
        return RandomSampler


class TestPeriodicSampler(TemplateTestRandom, unittest.TestCase):
    def get_query_strategy(self):
        return PeriodicSampler
