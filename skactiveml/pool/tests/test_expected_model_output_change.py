import unittest

import numpy as np

from skactiveml.pool import ExpectedModelOutputChange

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
    provide_test_regression_query_strategy_init_integration_dict,
    provide_test_regression_query_strategy_query_X_eval,
    provide_test_regression_query_strategy_change_dependence,
)
from skactiveml.regressor import NICKernelRegressor
from skactiveml.utils import MISSING_LABEL


class TestExpectedModelOutputChange(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 1, 2, MISSING_LABEL])
        self.reg = NICKernelRegressor()
        self.query_kwargs = dict(
            X=self.X,
            y=self.y,
            candidates=self.candidates,
            reg=self.reg,
        )

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, ExpectedModelOutputChange
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, ExpectedModelOutputChange
        )

    def test_init_param_loss(self):
        for poss_loss in [
            lambda x, y: np.average((x - y) ** 4),
            lambda x, y: np.average(np.abs(x - y)),
        ]:
            qs = ExpectedModelOutputChange(
                random_state=self.random_state, loss=poss_loss
            )
            qs.query(**self.query_kwargs)

        for illegal_loss in [lambda x: 0, "illegal"]:
            qs = ExpectedModelOutputChange(
                random_state=self.random_state, loss=illegal_loss
            )
            self.assertRaises(
                (TypeError, ValueError), qs.query, **self.query_kwargs
            )

    def test_init_param_integration_dict(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self, ExpectedModelOutputChange
        )

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, ExpectedModelOutputChange
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, ExpectedModelOutputChange
        )

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(
            self, ExpectedModelOutputChange, is_probabilistic=True
        )

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(
            self, ExpectedModelOutputChange
        )

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, ExpectedModelOutputChange
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, ExpectedModelOutputChange
        )

    def test_query_param_X_eval(self):
        provide_test_regression_query_strategy_query_X_eval(
            self, ExpectedModelOutputChange
        )

        qs = ExpectedModelOutputChange()
        y = np.full_like(self.y, 0)
        self.assertRaises(
            ValueError,
            qs.query,
            self.X,
            y,
            self.reg,
            candidates=self.candidates,
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, ExpectedModelOutputChange
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, ExpectedModelOutputChange
        )

    def test_logic(self):
        provide_test_regression_query_strategy_change_dependence(
            self, ExpectedModelOutputChange
        )
