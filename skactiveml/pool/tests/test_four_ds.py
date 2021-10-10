import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import CMM, SklearnClassifier
from skactiveml.pool import FourDS


class TestFourDS(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.X = StandardScaler().fit_transform(self.X)
        mixture_model = BayesianGaussianMixture(n_components=2)
        mixture_model.fit(self.X)
        self.clf = CMM(mixture_model=mixture_model)

    def test_init_param_lmbda(self):
        al4ds = FourDS(lmbda=True)
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=self.X, y=self.y)
        al4ds = FourDS(lmbda=1.1)
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=self.X, y=self.y)

    def test_query_param_clf(self):
        al4ds = FourDS()
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, clf=None,
                          X=self.X, y=self.y)
        clf = SklearnClassifier(GaussianProcessClassifier())
        al4ds = FourDS()
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, clf=clf,
                          X=self.X, y=self.y)

    def test_query_param_X(self):
        al4ds = FourDS()
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=None, y=self.y)
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=np.ones(5), y=self.y)
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=np.ones((5, 1)), y=self.y)

    def test_query_param_y(self):
        al4ds = FourDS()
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=self.X, y=None)
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=self.X, y=np.zeros((len(self.y), 2)))

    def test_query_param_sample_weight(self):
        al4ds = FourDS()
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=self.X, y=self.y, sample_weight=np.ones(1))
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, clf=self.clf,
                          X=self.X, y=self.y, sample_weight='test')

    def test_query(self):
        al4ds = FourDS(random_state=self.random_state)
        query_indices = al4ds.query(X_cand=self.X, clf=self.clf, X=self.X,
                                    y=self.y)
        self.assertEqual((1,), query_indices.shape)
        query_indices, utilities = al4ds.query(
            X_cand=self.X, X=self.X, clf=self.clf, y=self.y,
            return_utilities=True
        )
        self.assertEqual((1,), query_indices.shape)
        self.assertEqual((1, len(self.X)), utilities.shape)
        self.assertEqual(0, np.sum(utilities < 0))
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X,
                                               clf=self.clf, batch_size=3,
                                               y=self.y, return_utilities=True)
        self.assertEqual((3,), query_indices.shape)
        self.assertEqual((3, len(self.X)), utilities.shape)
        self.assertEqual(3, np.sum(np.isnan(utilities)))
        al4ds = FourDS(random_state=self.random_state)
        query_indices, utilities = al4ds.query(
            X_cand=self.X, X=self.X, clf=self.clf, batch_size=len(self.X) + 1,
            y=self.y, return_utilities=True
        )
        self.assertEqual((len(self.X),), query_indices.shape)
        self.assertEqual((len(self.X), len(self.X)), utilities.shape)
        self.assertEqual(np.sum(np.arange(0, len(self.X))),
                         np.sum(np.isnan(utilities)))
