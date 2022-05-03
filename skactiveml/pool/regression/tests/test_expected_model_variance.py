import unittest

from skactiveml.pool.regression import ExpectedModelVarianceMinimization
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
    provide_test_regression_query_strategy_init_integration_dict,
)


class TestExpectedModelVarianceMinimization(unittest.TestCase):
    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, ExpectedModelVarianceMinimization
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, ExpectedModelVarianceMinimization
        )

    def test_init_param_integration_dict(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(
            self, ExpectedModelVarianceMinimization, is_probabilistic=True
        )

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, ExpectedModelVarianceMinimization
        )
