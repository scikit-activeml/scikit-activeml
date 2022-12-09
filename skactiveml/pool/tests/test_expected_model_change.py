import numpy as np
import unittest

from copy import deepcopy
from scipy.stats import norm
from skactiveml.base import ProbabilisticRegressor
from skactiveml.pool import ExpectedModelChangeMaximization
from skactiveml.regressor import (
    NICKernelRegressor,
    SklearnRegressor,
    SklearnNormalRegressor,
)
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import is_unlabeled, call_func, MISSING_LABEL
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression


class TestExpectedModelChangeMaximization(
    TemplateSingleAnnotatorPoolQueryStrategy,
    unittest.TestCase,
):
    def setUp(self):
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
            "reg": SklearnRegressor(LinearRegression()),
        }
        super().setUp(
            qs_class=ExpectedModelChangeMaximization,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_bootstrap_size(self):
        test_cases = [
            (1, None),
            (-1, ValueError),
            ("five", TypeError),
            (0, ValueError),
        ]
        self._test_param("init", "bootstrap_size", test_cases)

    def test_init_param_n_train(self):
        test_cases = [
            (1, None),
            (0.0, ValueError),
            (1.5, ValueError),
            ("illegal", TypeError),
        ]
        self._test_param("init", "n_train", test_cases)

    def test_init_param_feature_map(self):
        test_cases = [
            ("illegal", TypeError),
            (1, TypeError),
            (lambda x: np.zeros((len(x), 1)), None),
        ]
        self._test_param("init", "feature_map", test_cases)

    def test_init_param_ord(self):
        test_cases = [
            (1, None),
            ("illegal", ValueError),
        ]
        self._test_param("init", "ord", test_cases)

    def test_query_param_reg(self):
        test_cases = [
            (NICKernelRegressor(), None),
            (SklearnNormalRegressor(GaussianProcessRegressor()), None),
            (GaussianProcessRegressor(), TypeError),
            (SklearnRegressor(GaussianProcessRegressor()), None),
        ]
        super().test_query_param_reg(test_cases=test_cases)

    def test_query(self):
        class ZeroRegressor(ProbabilisticRegressor):
            def predict_target_distribution(self, X):
                return norm(loc=np.zeros(len(X)))

            def fit(self, *args, **kwargs):
                return self

        qs = self.qs_class(**self.init_default_params)
        query_dict = deepcopy(self.query_default_params_reg)
        query_dict["reg"] = ZeroRegressor()
        query_dict["return_utilities"] = True
        utilities = call_func(qs.query, **query_dict)[1][0]
        np.testing.assert_almost_equal(
            np.zeros_like(query_dict["y"]),
            np.where(is_unlabeled(utilities), 0, utilities),
        )
