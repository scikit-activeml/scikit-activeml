import unittest

import numpy as np

from skactiveml.base import SkactivemlRegressor
from skactiveml.pool import RegressionTreeBasedAL
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL, is_labeled
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor


class TestRegressionTreeBasedAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.reg = SklearnRegressor(DecisionTreeRegressor(min_samples_leaf=2))

        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
            "reg": self.reg
        }
        super().setUp(
            qs_class=RegressionTreeBasedAL,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_method(self, test_cases=None):
        test_cases = test_cases or []
        test_cases += [(1, TypeError), ("string", ValueError)]
        self._test_param("init", "method", test_cases)

    def test_query_param_reg(self, test_cases=None):
        test_cases = test_cases or []
        test_cases += [
            (SklearnRegressor(NICKernelRegressor()), TypeError),
            (DecisionTreeRegressor(), TypeError)
        ]
        self._test_param("query", "reg", test_cases)

    def test_query(self):
        qs = self.qs_class()
        X = np.array([0, 2, 10, 12, 20, 22, 1, 11, 21]).reshape(-1, 1)
        y = np.append([0, 2, 10, 12, 20, 22], np.full(3, MISSING_LABEL))
        batch_size = 3

        idxs, utilities = qs.query(
            X, y, self.reg, batch_size=batch_size, return_utilities=True)
        self.reg.fit(X, y)
        np.testing.assert_array_equal(np.nansum(utilities[0]), batch_size)
        np.testing.assert_array_equal(utilities[0], np.append(6*[np.nan], 3*[1.]))

        # Method: 'random'
