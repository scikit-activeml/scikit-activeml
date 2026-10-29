import unittest
from unittest.mock import patch

import numpy as np
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    PairwiseKernel,
)
from sklearn.metrics import pairwise_kernels

from skactiveml.pool._quire import _one_versus_rest_transform, Quire
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import (
    MISSING_LABEL,
    is_labeled,
    simple_batch,
)


class TestQuire(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        self.candidates = np.array([1, 3])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
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
                self.assertTrue(
                    all(
                        len(call.args[0]) <= 256
                        for call in kernel.call_args_list
                    )
                )

    def test_query_matches_double_precision(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(25, 6)).astype(np.float32)
        y = np.arange(len(X), dtype=float) % 3
        y[-1] = np.nan
        for metric in ["rbf", "linear", "cosine", DotProduct(sigma_0=0)]:
            K = pairwise_kernels(X, X, metric=metric)
            for lmbda in [1e-4, 0.1, 1.0, 10.0]:
                for precomputed in [False, True]:
                    with self.subTest(
                        metric=metric, lmbda=lmbda, precomputed=precomputed
                    ):
                        qs = Quire(
                            [0, 1, 2],
                            lmbda=lmbda,
                            metric="precomputed" if precomputed else metric,
                        )
                        data = K if precomputed else X
                        _, utilities = qs.query(data, y, return_utilities=True)
                        _, expected = qs.query(
                            data.astype(float), y, return_utilities=True
                        )
                        np.testing.assert_array_equal(utilities, expected)

    def test_query_precomputed_kernel_dtypes(self):
        y = np.array([0, np.nan, np.nan])
        for K in [
            np.array([[100, 0, 20], [0, 200, 30], [20, 30, 250]], np.uint8),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], bool),
            np.array([[4e4, 0, 0], [0, 5e4, 1e4], [0, 1e4, 6e4]], np.float16),
        ]:
            with self.subTest(dtype=K.dtype):
                qs = Quire([0, 1], metric="precomputed")
                _, utilities = qs.query(K, y, return_utilities=True)
                _, expected = qs.query(
                    K.astype(float), y, return_utilities=True
                )
                np.testing.assert_array_equal(utilities, expected)

    def test_query_custom_kernels_keep_feature_dtype(self):
        X = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
        y = np.array([0, np.nan, np.nan])

        def kernel(x, z, gamma=None):
            return np.bitwise_and(x, z).sum()

        for metric in [
            kernel,
            PairwiseKernel(metric=kernel),
            PairwiseKernel(metric=kernel) + ConstantKernel(1),
        ]:
            with self.subTest(metric=metric):
                K = pairwise_kernels(X, metric=metric)
                expected = _full_quadratic_utilities(K, y, [0, 1], [1, 2], 1.0)
                _, utilities = Quire([0, 1], metric=metric).query(
                    X, y, return_utilities=True
                )
                np.testing.assert_allclose(utilities[0], expected)

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

    def test_query_symmetrizes_precomputed_kernel(self):
        K = np.array(
            [
                [0.8, 0.2, 0.1, 0.0, 0.6, 0.1],
                [0.1, 0.9, 0.3, 0.4, 0.1, 0.1],
                [0.5, 0.1, 0.9, 0.1, 0.1, 0.3],
                [0.1, 0.0, 0.3, 0.8, 0.2, 0.2],
                [0.2, 0.5, 0.1, 0.1, 0.5, 0.3],
                [0.1, 0.1, 0.2, 0.3, 0.2, 0.8],
            ]
        )
        y = np.array([np.nan, 0, np.nan, 1, np.nan, np.nan])
        expected = _full_quadratic_utilities(
            (K + K.T) / 2, y, [0, 1], [0, 2, 4, 5], 1.0
        )
        _, utilities = Quire([0, 1], metric="precomputed").query(
            K, y, return_utilities=True
        )
        np.testing.assert_allclose(utilities[0], expected)

    def test_query_candidate_subset_matches_direct_schur_complement(self):
        X = np.array(
            [
                [0, 0],
                [0.2, 0.1],
                [0.5, 0.4],
                [1.5, 0.8],
                [0.9, 1.1],
                [1.2, 0.3],
            ]
        )
        for classes in [[0, 1], [0, 1, 2]]:
            y = np.array([0, np.nan, 1, np.nan, classes[-1], np.nan])
            labeled = np.flatnonzero(~np.isnan(y))
            for metric in ["rbf", "linear", "precomputed"]:
                K = pairwise_kernels(
                    X, metric="rbf" if metric == "precomputed" else metric
                )
                for lmbda in [0.5, 2.0]:
                    L = np.linalg.inv(K + lmbda * np.eye(len(X)))
                    for candidates in [[5], [3, 5], [1, 3, 5]]:
                        with self.subTest(
                            classes=classes,
                            metric=metric,
                            lmbda=lmbda,
                            candidates=candidates,
                        ):
                            expected = np.full(len(X), np.nan)
                            for candidate in candidates:
                                known = np.append(labeled, candidate)
                                remaining = np.setdiff1d(
                                    np.arange(len(X)), known
                                )
                                # Minimize the full quadratic energy over the
                                # remaining unknown labels, without downdating
                                # an inverse or indexing the candidate list.
                                reduced = L[np.ix_(known, known)] - (
                                    L[np.ix_(known, remaining)]
                                    @ np.linalg.solve(
                                        L[np.ix_(remaining, remaining)],
                                        L[np.ix_(remaining, known)],
                                    )
                                )
                                energies = []
                                for label in classes:
                                    labels = np.append(
                                        np.where(y[labeled] == label, 1, -1), 1
                                    )
                                    energies.append(labels @ reduced @ labels)
                                expected[candidate] = -max(energies)
                            indices, utilities = Quire(
                                classes=classes,
                                metric=metric,
                                lmbda=lmbda,
                                random_state=0,
                            ).query(
                                K if metric == "precomputed" else X,
                                y,
                                candidates=candidates,
                                return_utilities=True,
                            )
                            np.testing.assert_allclose(utilities[0], expected)
                            self.assertEqual(
                                indices[0], np.nanargmax(expected)
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
