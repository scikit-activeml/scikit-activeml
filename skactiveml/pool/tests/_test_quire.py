import unittest

import numpy as np
from sklearn.metrics import pairwise_kernels

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool._quire import _del_i_inv, _L_aa_inv, \
    _one_versus_rest_transform, Quire
from skactiveml.utils import MISSING_LABEL, is_labeled, is_unlabeled


class TestQuire(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.y_MISSING_LABEL = np.array(
            [MISSING_LABEL, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL]
        )
        self.classes = np.array([0, 1])
        self.clf = ParzenWindowClassifier(
            classes=self.classes, random_state=self.random_state
        )
        self.kwargs = dict(candidates=self.candidates, X=self.X, y=self.y)
        self.kwargs_MISSING_LABEL = dict(
            candidates=self.candidates, X=self.X, y=self.y_MISSING_LABEL
        )

        lmbda = None,
        metric = "rbf",  # TODO default?
        metric_dict = None,

        clf,
        method = 0
    def test_query_param_clf(self):
        selector = UncertaintySampling()
        self.assertRaises(
            TypeError,
            selector.query,
            candidates=self.candidates,
            clf=GaussianProcessClassifier(),
            X=self.X,
            y=self.y,
        )

    def test_query_param_fit_clf(self):
        selector = Quire()
        self.assertRaises(
            TypeError, selector.query, **self.kwargs, fit_clf="string"
        )
        self.assertRaises(
            TypeError, selector.query, **self.kwargs, fit_clf=self.candidates
        )
        self.assertRaises(
            TypeError, selector.query, **self.kwargs, fit_clf=None
        )

    def test_query(self):
        pass

    def test__del_i_inv(self):
        A = np.random.random((3, 3))
        A_inv = np.linalg.inv(A)
        for i in range(len(A)):
            B = np.delete(np.delete(A, i, axis=0), i, axis=1)
            B_inv = np.linalg.inv(B)
            np.testing.assert_allclose(B_inv, _del_i_inv(A_inv, i))

    def test__L_aa_inv(self):
        lmbda = 1
        X = np.append(self.X, self.candidates, axis=0)
        y = np.append(self.y, np.full(len(self.candidates), MISSING_LABEL))
        is_lbld = is_labeled(y=y, missing_label=MISSING_LABEL)
        is_unlbld = is_unlabeled(y=y, missing_label=MISSING_LABEL)
        K = pairwise_kernels(X, X, metric='rbf')
        # compute L and L_aa
        L = np.linalg.inv(K + lmbda * np.eye(len(X)))
        L_aa = L[is_unlbld][:, is_unlbld]
        L_aa_inv = np.linalg.inv(L_aa)
        np.testing.assert_allclose(
            L_aa_inv, _L_aa_inv(K, lmbda, is_unlbld, is_lbld)
        )

    def test__one_versus_rest_transform(self):
        y = np.array([0, 1, 2, 1, 2, 0])
        y_ovr = np.array([[1, 0, 0, 0, 0, 1],
                          [0, 1, 0, 1, 0, 0],
                          [0, 0, 1, 0, 1, 0]]).T
        classes = np.unique(y)
        np.testing.assert_array_equal(
            y_ovr, _one_versus_rest_transform(y, classes, l_rest=0)
        )
