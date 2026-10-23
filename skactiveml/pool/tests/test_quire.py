import unittest

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
from skactiveml.utils import MISSING_LABEL, is_labeled, is_unlabeled


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

    def test_labeled_candidates_match_minimized_quadratic_energy(self):
        X = np.array([[0.0], [1.0], [2.5], [4.0], [5.5], [7.0]])
        classes = [-2, 7]
        partial_y = np.array([-2, 7, np.nan, -2, np.nan, np.nan])
        complete_y = np.array([-2, 7, 7, -2, -2, 7], dtype=float)
        kernel = pairwise_kernels(X, X, metric="rbf")
        lmbda = 0.7
        L = np.linalg.solve(kernel + lmbda * np.eye(len(X)), np.eye(len(X)))

        for metric in ["rbf", "precomputed"]:
            for y, candidates in [
                (partial_y, np.array([1])),
                (partial_y, np.array([5, 1, 4])),
                (complete_y, np.array([3, 1])),
            ]:
                with self.subTest(metric=metric, candidates=candidates):
                    expected = np.full(len(X), np.nan)
                    for candidate in candidates:
                        known = ~np.isnan(y)
                        known[candidate] = False
                        unknown = np.isnan(y)
                        unknown[candidate] = False
                        energies = []
                        for label in classes:
                            assignment = np.zeros(len(X))
                            assignment[known] = np.where(
                                y[known] == label, 1, -1
                            )
                            assignment[candidate] = 1
                            # Minimize the complete quadratic directly instead
                            # of reproducing Quire's block-inverse update.
                            assignment[unknown] = np.linalg.solve(
                                L[np.ix_(unknown, unknown)],
                                -L[np.ix_(unknown, ~unknown)]
                                @ assignment[~unknown],
                            )
                            energies.append(assignment @ L @ assignment)
                        expected[candidate] = -max(energies)

                    strategy = Quire(
                        classes, lmbda=lmbda, metric=metric, random_state=0
                    )
                    indices, utilities = strategy.query(
                        kernel if metric == "precomputed" else X,
                        y,
                        candidates=candidates,
                        batch_size=min(2, len(candidates)),
                        return_utilities=True,
                    )

                    self.assertTrue(np.isin(indices, candidates).all())
                    self.assertEqual(len(np.unique(indices)), len(indices))
                    for row, index in enumerate(indices):
                        np.testing.assert_allclose(utilities[row], expected)
                        self.assertAlmostEqual(
                            expected[index], np.nanmax(expected)
                        )
                        expected[index] = np.nan

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
