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
