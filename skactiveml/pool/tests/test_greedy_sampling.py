import unittest

import numpy as np

from skactiveml.base import SkactivemlRegressor
from skactiveml.pool import GreedySamplingX, GreedySamplingTarget
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL, is_labeled
from sklearn.gaussian_process import GaussianProcessRegressor


class TestGreedySamplingX(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
        }
        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
        }
        super().setUp(
            qs_class=GreedySamplingX,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_metric(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            (1.1, TypeError),
            ("euclidean", None),
        ]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ({}, None),
        ]
        self._test_param("init", "metric_dict", test_cases)

    def test_query(self):
        X = np.arange(7).reshape(7, 1)
        y = np.append([1], np.full(6, MISSING_LABEL))

        qs = GreedySamplingX()
        utilities = qs.query(X, y, return_utilities=True)[1][0]
        np.testing.assert_array_equal(
            utilities, np.append([MISSING_LABEL], np.arange(1, 7))
        )


class TestGreedySamplingTarget(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
            "reg": NICKernelRegressor(),
        }
        super().setUp(
            qs_class=GreedySamplingTarget,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_x_metric(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            (1.1, TypeError),
            ("euclidean", None),
        ]
        self._test_param("init", "x_metric", test_cases)

    def test_init_param_x_metric_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ({}, None),
        ]
        self._test_param("init", "x_metric_dict", test_cases)

    def test_init_param_y_metric(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            (1.1, TypeError),
            ("euclidean", None),
        ]
        self._test_param("init", "y_metric", test_cases)

    def test_init_param_y_metric_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ({}, None),
        ]
        self._test_param("init", "y_metric_dict", test_cases)

    def test_init_param_method(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ("GSy", None),
            ("GSi", None),
        ]
        self._test_param("init", "method", test_cases)

    def test_init_param_n_GSx_samples(self):
        test_cases = [
            (np.nan, TypeError),
            (1.5, TypeError),
            ({"test": 2}, TypeError),
            (0, None),
            (10, None),
        ]
        self._test_param("init", "n_GSx_samples", test_cases)

    def test_query_param_reg(self):
        test_cases = [
            (NICKernelRegressor(), None),
            (GaussianProcessRegressor(), TypeError),
            (SklearnRegressor(GaussianProcessRegressor()), None),
        ]
        super().test_query_param_reg(test_cases=test_cases)

    def test_query(self):
        X = (1 / 2 * np.arange(2 * 7) + 3.7).reshape(7, 2)
        y = [MISSING_LABEL, MISSING_LABEL, MISSING_LABEL, 0, 0, 0, 0]

        class ZeroRegressor(SkactivemlRegressor):
            def fit(self, *args, **kwargs):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        reg = ZeroRegressor()
        for method in ["GSy", "GSi"]:
            qs = GreedySamplingTarget(random_state=42, method=method)
            utilities = qs.query(X, y, reg, return_utilities=True)[1][0]
            np.testing.assert_array_equal(
                utilities, np.where(is_labeled(y), np.nan, 0)
            )
