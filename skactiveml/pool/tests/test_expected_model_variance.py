import unittest

from skactiveml.pool import ExpectedModelVarianceReduction
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


class TestExpectedModelVarianceMinimization(unittest.TestCase):
    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, ExpectedModelVarianceReduction
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, ExpectedModelVarianceReduction
        )

    def test_init_param_integration_dict(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(
            self, ExpectedModelVarianceReduction, is_probabilistic=True
        )

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_X_eval(self):
        provide_test_regression_query_strategy_query_X_eval(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, ExpectedModelVarianceReduction
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, ExpectedModelVarianceReduction
        )

    def test_logic(self):
        provide_test_regression_query_strategy_change_dependence(
            self, ExpectedModelVarianceReduction
        )
