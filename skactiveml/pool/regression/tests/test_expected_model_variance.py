import unittest

from skactiveml.pool.regression import ExpectedModelVarianceMinimization
from skactiveml.pool.regression.tests.test_pool_regression import (
    test_regression_query_strategy_init_random_state,
    test_regression_query_strategy_init_missing_label,
    test_regression_query_strategy_query_X,
    test_regression_query_strategy_query_y,
    test_regression_query_strategy_query_reg,
    test_regression_query_strategy_query_fit_reg,
    test_regression_query_strategy_query_sample_weight,
    test_regression_query_strategy_query_candidates,
    test_regression_query_strategy_query_batch_size,
    test_regression_query_strategy_query_return_utilities,
    test_regression_query_strategy_init_integration_dict,
)


class TestExpectedModelVarianceMinimization(unittest.TestCase):
    def test_init_param_random_state(self):
        test_regression_query_strategy_init_random_state(
            self, ExpectedModelVarianceMinimization
        )

    def test_init_param_missing_label(self):
        test_regression_query_strategy_init_missing_label(
            self, ExpectedModelVarianceMinimization
        )

    def test_init_param_integration_dict(self):
        test_regression_query_strategy_init_integration_dict(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_X(self):
        test_regression_query_strategy_query_X(self, ExpectedModelVarianceMinimization)

    def test_query_param_y(self):
        test_regression_query_strategy_query_y(self, ExpectedModelVarianceMinimization)

    def test_query_param_reg(self):
        test_regression_query_strategy_query_reg(
            self, ExpectedModelVarianceMinimization, is_probabilistic=True
        )

    def test_query_param_fit_reg(self):
        test_regression_query_strategy_query_fit_reg(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_sample_weight(self):
        test_regression_query_strategy_query_sample_weight(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_candidates(self):
        test_regression_query_strategy_query_candidates(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_batch_size(self):
        test_regression_query_strategy_query_batch_size(
            self, ExpectedModelVarianceMinimization
        )

    def test_query_param_return_utilities(self):
        test_regression_query_strategy_query_return_utilities(
            self, ExpectedModelVarianceMinimization
        )
