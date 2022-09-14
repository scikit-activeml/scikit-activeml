import unittest

import numpy as np

from skactiveml.pool import (
    KLDivergenceMaximization,
)
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


class TestKLDivergenceMaximization(unittest.TestCase):
    def setUp(self):
        self.random_state = 0

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, KLDivergenceMaximization
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, KLDivergenceMaximization
        )

    def test_init_param_integration_dict_potential_y_val(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self,
            KLDivergenceMaximization,
            integration_dict_name="integration_dict_target_val",
        )

    def test_init_param_integration_dict_cross_entropy(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self,
            KLDivergenceMaximization,
            integration_dict_name="integration_dict_cross_entropy",
        )

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, KLDivergenceMaximization
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, KLDivergenceMaximization
        )

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(
            self, KLDivergenceMaximization, is_probabilistic=True
        )

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(
            self, KLDivergenceMaximization
        )

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, KLDivergenceMaximization
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, KLDivergenceMaximization
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, KLDivergenceMaximization
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, KLDivergenceMaximization
        )

    def test_logic(self):
        provide_test_regression_query_strategy_change_dependence(
            self, KLDivergenceMaximization
        )

        qs = KLDivergenceMaximization()
        X = np.arange(5).reshape(5, 1)
        y = np.full(5, 0)
        X_cand = np.arange(3).reshape(3, 1)
        self.assertRaises(
            ValueError,
            qs.query,
            X,
            y,
            NICKernelRegressor(),
            candidates=X_cand,
        )
