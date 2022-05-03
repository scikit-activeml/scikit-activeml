import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.pool.regression import (
    MutualInformationGainMaximization,
    KLDivergenceMaximization,
    cross_entropy,
)
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
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor


class TestMutualInformationGainMaximization(unittest.TestCase):
    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, MutualInformationGainMaximization
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, MutualInformationGainMaximization
        )

    def test_init_param_integration_dict(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self, MutualInformationGainMaximization
        )

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, MutualInformationGainMaximization
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, MutualInformationGainMaximization
        )

    def test_query_param_reg(self):
        provide_test_regression_query_strategy_query_reg(
            self, MutualInformationGainMaximization, is_probabilistic=True
        )

    def test_query_param_fit_reg(self):
        provide_test_regression_query_strategy_query_fit_reg(
            self, MutualInformationGainMaximization
        )

    def test_query_param_sample_weight(self):
        provide_test_regression_query_strategy_query_sample_weight(
            self, MutualInformationGainMaximization
        )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, MutualInformationGainMaximization
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, MutualInformationGainMaximization
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, MutualInformationGainMaximization
        )


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
            integration_dict_name="integration_dict_potential_y_val",
        )

    def test_init_param_integration_dict_cross_entropy(self):
        provide_test_regression_query_strategy_init_integration_dict(
            self,
            KLDivergenceMaximization,
            integration_dict_name="integration_dict_cross_entropy",
        )

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(self, KLDivergenceMaximization)

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(self, KLDivergenceMaximization)

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

    def test_cross_entropy(self):
        X_1 = np.arange(3 * 2).reshape(3, 2)
        y_1 = np.arange(3, dtype=float) + 2
        X_2 = np.arange(5 * 2).reshape(5, 2)
        y_2 = 2 * np.arange(5, dtype=float) - 5
        reg_1 = NICKernelRegressor().fit(X_1, y_1)
        reg_2 = NICKernelRegressor().fit(X_2, y_2)

        result = cross_entropy(X_eval=X_1, true_reg=reg_1, other_reg=reg_2)
        self.assertEqual(y_1.shape, result.shape)

        for name, val in [
            ("X_eval", "illegal"),
            ("true_reg", SklearnRegressor(GaussianProcessRegressor()).fit(X_1, y_1)),
            ("other_reg", SklearnRegressor(GaussianProcessRegressor()).fit(X_1, y_1)),
            ("random_state", "illegal"),
            ("integration_dict", "illegal"),
        ]:
            cross_entropy_dict = dict(
                X_eval=X_1,
                true_reg=reg_1,
                other_reg=reg_2,
                random_state=self.random_state,
                integration_dict={},
            )
            cross_entropy_dict[name] = val
            self.assertRaises(
                (TypeError, ValueError), cross_entropy, **cross_entropy_dict
            )
