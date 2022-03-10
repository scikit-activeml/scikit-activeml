import unittest

import numpy as np
from sklearn.datasets import make_classification

from skactiveml.stream import PeriodicSampling, StreamRandomSampling


class TemplateTestStreamRandomSampling:
    def setUp(self):
        # initialise valid data to test uncertainty parameters
        rand = np.random.RandomState(0)
        stream_length = 100
        train_init_size = 10
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )

        self.X = X[[train_init_size], :]
        self.candidates = X[train_init_size:, :]
        self.y = y[:train_init_size]
        self.kwargs = dict(candidates=self.candidates)

    def test_init_param_budget(self):
        # budget must be defined as a float greater than 0
        query_strategy = self.get_query_strategy()(budget=[])
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = self.get_query_strategy()(budget="string")
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = self.get_query_strategy()(budget=-1)
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))

    def test_query_param_candidates(self):
        # candidates must be defined as a two dimensinal array
        query_strategy = self.get_query_strategy()()
        self.assertRaises(ValueError, query_strategy.query, candidates=1)
        self.assertRaises(ValueError, query_strategy.query, candidates=None)
        self.assertRaises(
            ValueError, query_strategy.query, candidates=np.ones(5)
        )

    def test_init_param_random_state(self):
        query_strategy = self.get_query_strategy()(
            random_state="string",
        )
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))

    def test_query_param_return_utilities(self):
        # return_utilities needs to be a boolean
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            candidates=self.candidates,
            return_utilities="string",
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            candidates=self.candidates,
            return_utilities=1,
        )

    def test_update_without_query(self):
        qs = self.get_query_strategy()()
        qs.update(np.array([[0], [1], [2]]), np.array([0, 2]))


class TestStreamRandomSampling(
    TemplateTestStreamRandomSampling, unittest.TestCase
):
    def get_query_strategy(self):
        return StreamRandomSampling

    def test_init_param_allow_exceeding_budget(self):
        # budget must be defined as a float greater than 0
        query_strategy = self.get_query_strategy()(
            allow_exceeding_budget="string"
        )
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = self.get_query_strategy()(allow_exceeding_budget=-1)
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))


class TestPeriodicSampling(
    TemplateTestStreamRandomSampling, unittest.TestCase
):
    def get_query_strategy(self):
        return PeriodicSampling
