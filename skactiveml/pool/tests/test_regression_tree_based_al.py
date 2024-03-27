import unittest

import numpy as np
from sklearn.metrics import pairwise_distances_argmin

from skactiveml.pool import RegressionTreeBasedAL
from skactiveml.pool._regression_tree_based_al import (
    _calc_acquisitions_per_leaf,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL, is_unlabeled
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


class TestRegressionTreeBasedAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.reg = SklearnRegressor(DecisionTreeRegressor(min_samples_leaf=2))

        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
            "reg": self.reg,
        }
        super().setUp(
            qs_class=RegressionTreeBasedAL,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_method(self, test_cases=None):
        test_cases = test_cases or []
        test_cases += [
            (1, TypeError),
            ("string", ValueError),
            ("random", None),
            ("diversity", None),
            ("representativity", None),
        ]
        self._test_param("init", "method", test_cases)

    def test_init_param_max_iter_representativity(self, test_cases=None):
        test_cases = test_cases or []
        test_cases += [
            (-1, ValueError),
            ("string", TypeError),
            (1, None),
            (10, None),
        ]
        self._test_param(
            "init",
            "max_iter_representativity",
            test_cases,
            replace_init_params={"method": "representativity"},
        )

    def test_query_param_reg(self, test_cases=None):
        test_cases = test_cases or []
        test_cases += [
            (SklearnRegressor(NICKernelRegressor()), TypeError),
            (DecisionTreeRegressor(), TypeError),
            (SklearnRegressor(DecisionTreeRegressor()), None),
            (SklearnRegressor(ExtraTreeRegressor()), None),
        ]
        self._test_param("query", "reg", test_cases)

    def test__calc_acquisitions_per_leaf(self):
        reg = SklearnRegressor(_DummyRegressor())
        X = np.array([0, 2, 10, 12, 20, 22, 1, 11, 21]).reshape(-1, 1)
        y = np.append([0, 2, 10, 12, 20, 22], np.full(3, MISSING_LABEL))
        np.testing.assert_allclose(
            _calc_acquisitions_per_leaf(X, y, reg, MISSING_LABEL),
            np.full(3, 1 / 3),
        )

    def test_query(self):
        qs = self.qs_class(random_state=0)
        X = np.array([0, 2, 10, 12, 20, 22, 1, 11, 21]).reshape(-1, 1)
        y = np.append([0, 2, 10, 12, 20, 22], np.full(3, MISSING_LABEL))
        batch_size = 3

        idxs, utilities = qs.query(
            X, y, self.reg, batch_size=batch_size, return_utilities=True
        )
        self.reg.fit(X, y)
        np.testing.assert_array_equal(
            utilities[0], np.append(6 * [np.nan], 3 * [1.0])
        )

        # Method: 'representativity'
        delta = np.array([1, 1, 1])
        R = np.array([0, 0, 0])
        utils_expected = np.full((batch_size, len(X)), np.nan)
        utils_expected[0, 6] = (delta - R)[0]
        utils_expected[1, 7] = (delta - R)[1]
        utils_expected[2, 8] = (delta - R)[2]

        qs = self.qs_class(
            method="representativity", max_iter_representativity=1
        )
        reg = SklearnRegressor(_DummyRegressor())
        _, utils = qs.query(
            X, y, reg, batch_size=batch_size, return_utilities=True
        )
        np.testing.assert_allclose(utils_expected, utils)
        qs.query(X, y, reg, batch_size=batch_size)
        qs.query(X, y, reg, candidates=[[1]], batch_size=batch_size)

        batch_size = 1
        y[1] = MISSING_LABEL
        y[3] = MISSING_LABEL
        y[5] = MISSING_LABEL
        delta = np.array([np.nan, 2, np.nan, 2, np.nan, 2, 1, 1, 1])
        R = np.array([np.nan, 1, np.nan, 1, np.nan, 1, 1, 1, 1])
        utils_expected = np.full((batch_size, len(X)), np.nan)
        utils_expected[0, 1] = (delta - R)[1]
        utils_expected[0, 6] = (delta - R)[6]

        qs = self.qs_class(
            method="representativity",
            max_iter_representativity=1,
            random_state=0,
        )
        reg = SklearnRegressor(_DummyRegressor())
        _, utils = qs.query(
            X, y, reg, batch_size=batch_size, return_utilities=True
        )
        np.testing.assert_allclose(utils_expected, utils)

        # Method: 'diversity'
        X = np.linspace(0, 100, 101).reshape(-1, 1)
        y = np.full(len(X), MISSING_LABEL)
        y[50] = 50
        utils_expected = (
            np.abs(X - 50).flatten() * batch_size / np.sum(is_unlabeled(y))
        )
        utils_expected[50] = MISSING_LABEL

        qs = self.qs_class(method="diversity")
        _, utilities = qs.query(
            X, y, self.reg, batch_size=batch_size, return_utilities=True
        )
        np.testing.assert_allclose(utils_expected, utilities[0])
        qs.query(X, y, self.reg, candidates=[[1]], batch_size=batch_size)

        # Method: 'random'
        qs = self.qs_class(method="random")
        qs.query(X, y, self.reg, batch_size=batch_size)
        qs.query(X, np.full_like(y, np.nan), self.reg)
        qs.query(X, np.full_like(y, np.nan), self.reg, candidates=[[1]])


class _DummyRegressor(DecisionTreeRegressor):
    centers = np.array([1, 11, 21]).reshape(-1, 1)
    node_count = 3

    def apply(self, X):
        return pairwise_distances_argmin(X, self.centers, axis=1)

    def __getattr__(self, item):
        if item == "tree_":
            return self
        raise AttributeError
