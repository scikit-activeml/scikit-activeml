import unittest

import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_regression

from skactiveml.pool import RegressionTreeBasedAL
from skactiveml.pool._regression_tree_based_al import (
    _calc_acquisitions_per_leaf,
    _discretize_acquisitions_per_leaf,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


class TestRegressionTreeBasedAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.reg = SklearnRegressor(
            DecisionTreeRegressor(min_samples_leaf=2, random_state=0)
        )

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

    def test__discretize_acquisitions_per_leaf(self):
        n_k = np.array([2.5, 4.0, 3.9, 7.3, 9.6])
        n_k_discrete = _discretize_acquisitions_per_leaf(
            n_k, np.round(n_k.sum()).astype(int), np.random.RandomState(0)
        )
        # Ensures the correct `batch_size`, i.e., number of acquisitions.
        self.assertEqual(n_k_discrete.sum(), np.floor(n_k.sum()))

        # Ensures the correct minimum acquisitions per leaf.
        self.assertTrue((np.abs(n_k_discrete - n_k) <= 1).all())

        # Checks reproducibility.
        for _ in range(5):
            n_k_discrete_new = _discretize_acquisitions_per_leaf(
                n_k, np.round(n_k.sum()).astype(int), np.random.RandomState(0)
            )
            np.testing.assert_array_equal(n_k_discrete, n_k_discrete_new)

        # Checks that different random states can lead to different results.
        n_k = np.array([0.9] * 100)
        n_k_discrete = _discretize_acquisitions_per_leaf(
            n_k, np.round(n_k.sum()).astype(int), np.random.RandomState(0)
        )
        n_k_discrete_new = _discretize_acquisitions_per_leaf(
            n_k, np.round(n_k.sum()).astype(int), np.random.RandomState(2)
        )
        self.assertTrue((n_k_discrete != n_k_discrete_new).any())

    def test_query(self):
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = np.sin(X.ravel())
        y[30:70] = MISSING_LABEL
        batch_size = 10

        # Labels to test fallback to random sampling.
        y_one_label = np.full_like(y, fill_value=np.nan)
        y_one_label[0] = 1

        # Test varying methods.
        for method in ["diversity", "representativity"]:
            for candidates in [None, range(44, 56), X[range(44, 56)]]:
                qs = self.qs_class(random_state=0, method=method)
                idxs, utilities = qs.query(
                    X,
                    y,
                    self.reg,
                    batch_size=batch_size,
                    return_utilities=True,
                    candidates=candidates,
                )
                self.reg.fit(X, y)
                u_neg_inf = np.isneginf(utilities)
                u_neg_inf_sum = u_neg_inf.sum(axis=1)
                if method in ["random", "diversity"]:
                    u_method_test = (
                        utilities == 1
                        if method == "random"
                        else utilities >= 0
                    )
                    if u_method_test is not None:
                        self.assertTrue(
                            (
                                u_neg_inf + np.isnan(utilities) + u_method_test
                            ).all()
                        )

                    if candidates is None:
                        self.assertTrue(
                            (
                                (10 <= u_neg_inf_sum) & (u_neg_inf_sum <= 20)
                            ).all()
                        )
                    else:
                        self.assertTrue(
                            ((0 <= u_neg_inf_sum) & (u_neg_inf_sum <= 6)).all()
                        )
                else:
                    n_candidates = (-np.inf < utilities).sum()
                    if candidates is None:
                        self.assertEqual(n_candidates, 40)
                    else:
                        self.assertEqual(n_candidates, 12)

                # Test fallback to random sampling.
                qs = self.qs_class(random_state=0, method=method)
                idxs, utilities = qs.query(
                    X,
                    y_one_label,
                    self.reg,
                    batch_size=batch_size,
                    return_utilities=True,
                    candidates=candidates,
                )
                self.assertTrue((np.isnan(utilities) + (utilities == 1)).all())

        # reg = SklearnRegressor(DecisionTreeRegressor(), missing_label=np.nan)
        X, y = make_regression(n_samples=500, n_features=100, random_state=0)
        y[100:] = np.nan
        query_indices = []
        for i in [1, 10]:
            qs = RegressionTreeBasedAL(
                method="representativity",
                max_iter_representativity=i,
                random_state=0,
            )
            query_indices.append(qs.query(X, y, reg=self.reg, batch_size=5))
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            query_indices[0],
            query_indices[1],
        )


class _DummyRegressor(DecisionTreeRegressor):
    centers = np.array([1, 11, 21]).reshape(-1, 1)
    node_count = 3

    def apply(self, X):
        return pairwise_distances_argmin(X, self.centers, axis=1)

    def __getattr__(self, item):
        if item == "tree_":
            return self
        raise AttributeError
