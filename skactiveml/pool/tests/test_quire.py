import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import pairwise_kernels

from skactiveml.pool._quire import (
    _del_i_inv,
    _L_aa_inv,
    _one_versus_rest_transform,
    Quire,
)
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import assert_no_query_state
from skactiveml.utils import (
    MISSING_LABEL,
    is_labeled,
    is_unlabeled,
    simple_batch,
)


class TestQuire(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        self.candidates = np.array([1, 3])
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y_true = np.array([0, 0, 1, 1])
        self.y = np.array([0, MISSING_LABEL, 1, MISSING_LABEL])
        self.classes = np.array([0, 1])
        self.kwargs = dict(
            candidates=self.candidates,
            X=self.X,
            y=self.y,
        )
        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
        }
        super().setUp(
            qs_class=Quire,
            init_default_params={"classes": [0, 1]},
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(None, TypeError), (Quire, TypeError)]
        self._test_param("init", "classes", test_cases)
        self._test_param("init", "classes", [([0, 1], None)])
        self._test_param(
            "init",
            "classes",
            [(["0", "1"], None)],
            {"missing_label": "none"},
            {"y": ["0", "1", "none", "none"]},
        )

    def test_nested_classes_resolve_before_capability_rejection(self):
        strategy = Quire(classes=[[0, 1, 2], [0, 1, 2]])
        y = np.array([[0, 1], [1, 2], [np.nan, np.nan], [np.nan, np.nan]])

        with self.assertRaisesRegex(
            ValueError,
            "Quire does not support target capability.*multi-output",
        ):
            strategy.query(X=self.X, y=y)

        assert_no_query_state(self, strategy)

    def test_init_param_lmbda(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (-1, ValueError),
            (0, ValueError),
            ("string", TypeError),
        ]
        self._test_param("init", "lmbda", test_cases)

    def test_init_param_metric_dict(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (42, TypeError),
            ({"string": None}, TypeError),
            ("string", TypeError),
        ]
        self._test_param("init", "metric_dict", test_cases)

    def test_init_param_metric(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(42, ValueError), ("string", ValueError)]
        self._test_param("init", "metric", test_cases)

        K = np.zeros((len(self.y), len(self.y) - 1))
        test_cases += [("precomputed", ValueError)]
        self._test_param(
            "init", "metric", test_cases, replace_query_params={"X": K}
        )

    def test_query(self):
        # Test metric="precomputed"
        qs = Quire(self.classes, metric="precomputed")
        K = pairwise_kernels(self.X, self.X, metric="rbf")
        _, utils = qs.query(K, self.y, return_utilities=True)
        qs = Quire(self.classes, metric="rbf")
        _, expected_utils = qs.query(**self.kwargs, return_utilities=True)
        np.testing.assert_array_equal(expected_utils, utils)

        # Test with zero labels.
        qs.query(X=self.X, y=np.full(shape=len(self.X), fill_value=np.nan))

        # Test Scenario.
        qs = Quire(self.classes, metric="precomputed")
        K = np.zeros_like(K)
        _, utils = qs.query(K, self.y, return_utilities=True)
        is_lbld = is_labeled(self.y)
        y_labeled = self.y[is_lbld].reshape(-1, 1) * 2 - 1
        expected_utils = np.full_like(utils, -1 - y_labeled.T.dot(y_labeled))
        np.testing.assert_array_equal(
            expected_utils[:, ~is_lbld], utils[:, ~is_lbld]
        )

        qs = Quire(self.classes)
        _, utils = qs.query(**self.kwargs, return_utilities=True)

    def test_query_matches_full_quadratic_objective(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(size=(19, 4))
        # Include kernels that are not positive semidefinite, and nonconstant
        # kernel diagonals, without relying on the optimized scoring identity.
        metrics = [
            ("rbf", {"gamma": 0.3}),
            ("linear", {}),
            ("poly", {"degree": 2, "coef0": 0.2}),
            ("sigmoid", {"gamma": 0.2}),
            ("cosine", {}),
            ("laplacian", {}),
            ("chi2", {}),
            ("additive_chi2", {}),
            (lambda x, y: np.dot(x, y) - 2, {}),
        ]
        for n_labeled in [0, 1, 7, 18]:
            y = np.full(len(X), np.nan)
            labeled = rng.choice(len(X), n_labeled, replace=False)
            y[labeled] = np.arange(n_labeled) % 3
            candidates = np.flatnonzero(np.isnan(y))
            for metric, params in metrics:
                K = pairwise_kernels(X, X, metric=metric, **params)
                for mapping in [candidates, candidates[::-2]]:
                    for precomputed in [False, True]:
                        with self.subTest(
                            n_labeled=n_labeled,
                            metric=metric,
                            mapping=mapping,
                            precomputed=precomputed,
                        ):
                            lmbda = 2.0
                            expected = _full_quadratic_utilities(
                                K, y, [0, 1, 2], mapping, lmbda
                            )
                            qs = Quire(
                                classes=[0, 1, 2],
                                metric=(
                                    "precomputed" if precomputed else metric
                                ),
                                metric_dict=None if precomputed else params,
                                lmbda=lmbda,
                                random_state=42,
                            )
                            indices, utilities = qs.query(
                                K if precomputed else X,
                                y,
                                candidates=mapping,
                                batch_size=min(3, len(mapping)),
                                return_utilities=True,
                            )
                            for i, index in enumerate(indices):
                                np.testing.assert_allclose(
                                    utilities[i],
                                    expected,
                                    rtol=1e-9,
                                    atol=1e-9,
                                )
                                self.assertAlmostEqual(
                                    expected[index], np.nanmax(expected)
                                )
                                expected[index] = np.nan

    def test_query_kernel_diagonal_batches(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(size=(600, 3))
        X[0] = 0
        y = np.full(len(X), np.nan)
        for metric in ["rbf", "linear", "cosine", "poly"]:
            with self.subTest(metric=metric):
                K = pairwise_kernels(X, metric=metric)
                expected = -1 / (np.diag(K) + 1)
                with patch(
                    "skactiveml.pool._quire.pairwise_kernels",
                    wraps=pairwise_kernels,
                ) as kernel:
                    _, utilities = Quire([0, 1], metric=metric).query(
                        X, y, return_utilities=True
                    )
                np.testing.assert_allclose(utilities[0], expected)
                # With no labels, only bounded diagonal blocks are needed.
                self.assertTrue(
                    all(
                        len(call.args[0]) <= 256
                        for call in kernel.call_args_list
                    )
                )

    def test_query_single_precision_kernel_compatibility(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(25, 6)).astype(np.float32)
        y = np.arange(len(X), dtype=float) % 3
        y[-1] = np.nan
        for metric in ["rbf", "linear", "cosine"]:
            K = pairwise_kernels(X, X, metric=metric)
            for lmbda in [1e-4, 0.1, 1.0, 10.0]:
                expected = _full_quadratic_utilities(
                    K, y, [0, 1, 2], [len(X) - 1], lmbda
                )
                for precomputed in [False, True]:
                    with self.subTest(
                        metric=metric, lmbda=lmbda, precomputed=precomputed
                    ):
                        _, utilities = Quire(
                            [0, 1, 2],
                            lmbda=lmbda,
                            metric="precomputed" if precomputed else metric,
                        ).query(
                            K if precomputed else X, y, return_utilities=True
                        )
                        np.testing.assert_allclose(
                            utilities[0], expected, rtol=1e-8, atol=1e-8
                        )

    def test_query_rejects_labeled_candidates(self):
        for candidates in [[0], [0, 3], [2, 1, 3]]:
            with self.subTest(candidates=candidates):
                with self.assertRaisesRegex(ValueError, "labeled samples"):
                    Quire([0, 1]).query(self.X, self.y, candidates=candidates)

    def test_query_ties_and_candidate_order(self):
        X = np.zeros((9, 2))
        y = np.full(9, np.nan)
        y[2] = 0
        candidates = np.array([8, 5, 1, 3])
        qs = Quire([0, 1], metric="linear", random_state=42)
        actual = qs.query(
            X, y, candidates=candidates, batch_size=4, return_utilities=True
        )
        expected = np.full(len(X), np.nan)
        expected[candidates] = -2
        qs._validate_data(X, y, candidates, 4, True, reset=True)
        reference = simple_batch(
            expected, qs.random_state_, batch_size=4, return_utilities=True
        )
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[1], reference[1])

    def test_query_asymmetric_callable_preserves_symmetrization(self):
        X = np.arange(18).reshape(6, 3) / 10
        y = np.array([np.nan, 0, np.nan, 1, np.nan, np.nan])

        def kernel(x, z):
            return np.dot(x, z) + x[0] - z[0]

        K = pairwise_kernels(X, X, metric=kernel)
        candidates = np.array([5, 0, 4])
        expected = _full_quadratic_utilities(K, y, [0, 1], candidates, 1.0)
        _, utilities = Quire([0, 1], metric=kernel).query(
            X, y, candidates=candidates, return_utilities=True
        )
        np.testing.assert_allclose(utilities[0], expected)

    def test_query_nonsymmetric_precomputed_compatibility(self):
        K = np.array(
            [
                [0.8, 0.2, 0.1, 0.0, 0.6, 0.1],
                [0.1, 0.9, 0.3, 0.0, 0.1, 0.1],
                [0.5, 0.1, 0.9, 0.1, 0.1, 0.3],
                [0.1, 0.0, 0.3, 0.8, 0.2, 0.2],
                [0.2, 0.5, 0.1, 0.1, 0.5, 0.3],
                [0.1, 0.1, 0.2, 0.3, 0.2, 0.8],
            ]
        )
        y = np.array([np.nan, 0, np.nan, 1, np.nan, np.nan])
        # Scores produced by the original inverse/downdate implementation.
        expected = [
            -1.62206740,
            np.nan,
            -1.58640760,
            np.nan,
            -1.99950640,
            -1.70899301,
        ]
        with self.assertWarnsRegex(UserWarning, "not symmetric"):
            _, utilities = Quire([0, 1], metric="precomputed").query(
                K, y, return_utilities=True
            )
        np.testing.assert_allclose(utilities[0], expected, atol=1e-8)

    def test__del_i_inv(self):
        self.assertWarns(Warning, _del_i_inv, np.tri(5), 2)

        A = np.random.random((3, 3))
        A = A + A.T
        A_inv = np.linalg.inv(A)
        for i in range(len(A)):
            B = np.delete(np.delete(A, i, axis=0), i, axis=1)
            B_inv = np.linalg.inv(B)
            np.testing.assert_allclose(B_inv, _del_i_inv(A_inv, i))

    def test__L_aa_inv(self):
        lmbda = 1
        X = np.append(self.X, self.X_cand, axis=0)
        y = np.append(self.y_true, np.full(len(self.X_cand), MISSING_LABEL))
        is_lbld = is_labeled(y=y, missing_label=MISSING_LABEL)
        is_unlbld = is_unlabeled(y=y, missing_label=MISSING_LABEL)
        K = pairwise_kernels(X, X, metric="rbf")
        # compute L and L_aa
        L = np.linalg.inv(K + lmbda * np.eye(len(X)))
        L_aa = L[is_unlbld][:, is_unlbld]
        L_aa_inv = np.linalg.inv(L_aa)
        np.testing.assert_allclose(
            L_aa_inv, _L_aa_inv(K, lmbda, is_unlbld, is_lbld)
        )

    def test__one_versus_rest_transform(self):
        y = np.array([0, 1, 2, 1, 2, 0])
        y_ovr = np.array(
            [[1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0]]
        ).T
        classes = np.unique(y)
        np.testing.assert_array_equal(
            y_ovr, _one_versus_rest_transform(y, classes, l_rest=0)
        )


def _full_quadratic_utilities(K, y, classes, candidates, lmbda):
    """Evaluate the relaxed objective directly using the full precision."""
    precision = np.linalg.inv(K + lmbda * np.eye(len(K)))
    labeled = np.flatnonzero(~np.isnan(y))
    targets = np.where(y[labeled, None] == classes, 1.0, -1.0)
    utilities = np.full(len(K), np.nan)
    for candidate in candidates:
        fixed = np.append(labeled, candidate)
        free = np.setdiff1d(np.arange(len(K)), fixed)
        values = np.vstack((targets, np.ones(len(classes))))
        coupling = precision[np.ix_(free, fixed)] @ values
        objective = values.T @ precision[np.ix_(fixed, fixed)] @ values
        objective -= coupling.T @ np.linalg.solve(
            precision[np.ix_(free, free)], coupling
        )
        utilities[candidate] = -np.diag(objective).max()
    return utilities
