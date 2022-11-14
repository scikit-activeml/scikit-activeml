import numpy as np
import unittest

from copy import deepcopy
from scipy.stats import norm
from skactiveml.base import ProbabilisticRegressor
from skactiveml.pool import ExpectedModelOutputChange
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


class TestExpectedModelOutputChange(
    TemplateSingleAnnotatorPoolQueryStrategy,
    unittest.TestCase,
):
    def setUp(self):
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
            "reg": NICKernelRegressor(),
        }
        super().setUp(
            qs_class=ExpectedModelOutputChange,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_loss(self):
        test_cases = [
            (lambda x, y: np.average((x - y) ** 4), None),
            (lambda x, y: np.average(np.abs(x - y)), None),
            (lambda x: 0, ValueError),
            ("illegal", TypeError),
        ]
        self._test_param("init", "loss", test_cases)

    def test_init_param_integration_dict(self):
        test_cases = [
            ({"method": "assume_linear"}, None),
            ({"method": "monte_carlo"}, None),
            ({}, None),
            ({"method": "illegal"}, TypeError),
            ("illegal", TypeError),
        ]
        self._test_param("init", "integration_dict", test_cases)

    def test_query_param_reg(self):
        test_cases = [
            (NICKernelRegressor(), None),
            (SklearnNormalRegressor(GaussianProcessRegressor()), None),
            (GaussianProcessRegressor(), TypeError),
            (SklearnRegressor(GaussianProcessRegressor()), TypeError),
        ]
        super().test_query_param_reg(test_cases=test_cases)

    def test_query_param_X_eval(self):
        test_cases = [
            (None, None),
            (np.array([[1, 2], [5, 8]]), None),
            (np.arange(3), ValueError),
            ("illegal", ValueError),
        ]
        self._test_param("query", "X_eval", test_cases)
        test_cases = [(None, ValueError)]
        query_params = deepcopy(self.query_default_params_reg)
        query_params["y"] = np.ones_like(self.query_default_params_reg["y"])
        self._test_param(
            "query",
            "X_eval",
            test_cases=test_cases,
            replace_query_params=query_params,
        )

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
