import unittest

import numpy as np

from skactiveml.pool.regression import GreedySamplingX, GreedySamplingY
from skactiveml.pool.regression.tests.provide_test_pool_regression import (
    provide_test_regression_query_strategy_init_random_state,
    provide_test_regression_query_strategy_init_missing_label,
    provide_test_regression_query_strategy_query_X,
    provide_test_regression_query_strategy_query_y,
    provide_test_regression_query_strategy_query_reg,
    provide_test_regression_query_strategy_query_fit_reg,
    provide_test_regression_query_strategy_query_sample_weight,
    provide_test_regression_query_strategy_query_candidates,
    provide_test_regression_query_strategy_query_batch_size,
    provide_test_regression_query_strategy_query_return_utilities,
)
from skactiveml.regressor import NICKernelRegressor
from skactiveml.utils import MISSING_LABEL


class TestGreedySamplingX(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 1, 2, -2])
        self.query_kwargs = dict(X=self.X, y=self.y, candidates=self.candidates)

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(self, GreedySamplingX)

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(self, GreedySamplingX)

    def test_init_param_metric(self):
        qs = GreedySamplingX(metric="illegal", random_state=self.random_state)
        self.assertRaises(ValueError, qs.query, **self.query_kwargs)

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(self, GreedySamplingX)

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(self, GreedySamplingX)

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(self, GreedySamplingX)

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(self, GreedySamplingX)

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, GreedySamplingX
        )


class TestGreedySamplingY(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 1, 2, -2])
        self.reg = NICKernelRegressor()
        self.query_kwargs = dict(
            X=self.X, y=self.y, candidates=self.candidates, reg=self.reg
        )

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(self, GreedySamplingY)

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(self, GreedySamplingY)

    def test_init_param_x_metric(self):
        self.query_kwargs["y"] = np.full(len(self.y), MISSING_LABEL)
        qs = GreedySamplingY(x_metric="illegal", random_state=self.random_state)
        self.assertRaises(ValueError, qs.query, **self.query_kwargs)

    def test_init_param_y_metric(self):
        qs = GreedySamplingY(y_metric="illegal", random_state=self.random_state)
        self.assertRaises(ValueError, qs.query, **self.query_kwargs)

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(self, GreedySamplingY)

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(self, GreedySamplingY)

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(self, GreedySamplingY)

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(self, GreedySamplingY)

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, GreedySamplingY
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(self, GreedySamplingY)

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(self, GreedySamplingY)

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, GreedySamplingY
        )
