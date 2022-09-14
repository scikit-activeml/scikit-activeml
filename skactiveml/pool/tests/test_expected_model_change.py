import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from skactiveml.pool import ExpectedModelChangeMaximization
from skactiveml.pool.tests.provide_test_pool_regression import (
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
    provide_test_regression_query_strategy_change_dependence,
)
from skactiveml.regressor import SklearnRegressor


class TestExpectedModelChangeMaximization(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4], [3.5, 2], [4.2, 4]])
        self.y = np.array([np.nan, np.nan, 2, -2, 3.4, 2.7])
        self.reg = SklearnRegressor(LinearRegression())
        self.qs = ExpectedModelChangeMaximization()
        self.query_kwargs = dict(
            X=self.X, y=self.y, candidates=self.candidates, reg=self.reg
        )

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, ExpectedModelChangeMaximization
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, ExpectedModelChangeMaximization
        )

    def test_init_param_k_bootstrap(self):
        for wrong_val, error in zip(["five", 0], [TypeError, ValueError]):
            qs = ExpectedModelChangeMaximization(bootstrap_size=wrong_val)
            self.assertRaises(error, qs.query, **self.query_kwargs)

    def test_init_param_n_train(self):
        for wrong_val, error in zip(["five", 1.5], [TypeError, ValueError]):
            qs = ExpectedModelChangeMaximization(n_train=wrong_val)
            self.assertRaises(error, qs.query, **self.query_kwargs)

    def test_init_param_feature_map(self):
        for wrong_val in ["wrong_val", 1]:
            qs = ExpectedModelChangeMaximization(feature_map=wrong_val)
            self.assertRaises(TypeError, qs.query, **self.query_kwargs)

        qs = ExpectedModelChangeMaximization(
            feature_map=lambda x: np.zeros((len(x), 1)),
            random_state=self.random_state,
        )
        utilities = qs.query(
            self.X, self.y, reg=self.reg, return_utilities=True, fit_reg=True
        )[1]
        np.testing.assert_array_equal(np.zeros(2), utilities[0, :2])

    def test_init_param_ord(self):
        qs = ExpectedModelChangeMaximization(ord="wrong_norm")
        self.assertRaises(ValueError, qs.query, **self.query_kwargs)

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, ExpectedModelChangeMaximization
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, ExpectedModelChangeMaximization
        )

    def test_logic(self):
        provide_test_regression_query_strategy_change_dependence(
            self, ExpectedModelChangeMaximization
        )
