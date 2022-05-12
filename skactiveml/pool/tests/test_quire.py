import unittest

import numpy as np
from sklearn.metrics import pairwise_kernels

from skactiveml.pool._quire import (
    _del_i_inv,
    _L_aa_inv,
    _one_versus_rest_transform,
    Quire,
)
from skactiveml.utils import MISSING_LABEL, is_labeled, is_unlabeled


class TestQuire(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
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

    def test_init_param_classes(self):
        qs = Quire(self.classes)
        self.assertTrue(hasattr(qs, "classes"))

    def test_init_param_lmbda(self):
        for lmbda in [-1, 0, "string"]:
            qs = Quire(self.classes, lmbda=lmbda)
            self.assertRaises((ValueError, TypeError), qs.query, **self.kwargs)

    def test_init_param_metric_dict(self):
        for metric_dict in ["String", 42, {"string": None}]:
            qs = Quire(self.classes, metric_dict=metric_dict)
            self.assertRaises(TypeError, qs.query, **self.kwargs)

    def test_init_param_metric(self):
        qs = Quire(self.classes, metric="Test")
        self.assertRaises(ValueError, qs.query, **self.kwargs)
        qs = Quire(self.classes, metric=42)
        self.assertRaises(ValueError, qs.query, **self.kwargs)
        qs = Quire(self.classes, metric="precomputed")
        K = np.zeros((len(self.y), len(self.y) - 1))
        self.assertRaises(ValueError, qs.query, y=self.y, X=K)

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
