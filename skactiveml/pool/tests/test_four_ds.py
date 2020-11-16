import numpy as np
import unittest

from skactiveml.pool import FourDS
from skactiveml.classifier import CMM
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


class TestFourDS(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.X = StandardScaler().fit_transform(self.X)
        mixture_model = BayesianGaussianMixture(n_components=2)
        mixture_model.fit(self.X)
        self.CMM = CMM(mixture_model=mixture_model)

    def test_fit(self):
        al4ds = FourDS(clf=CMM(), random_state=self.random_state)
        self.assertTrue(isinstance(al4ds.clf, CMM))
        self.assertEqual(al4ds.lmbda, None)
        self.assertEqual(al4ds.random_state, self.random_state)

    def test_query(self):
        al4ds = FourDS(clf=GaussianProcessClassifier())
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)
        al4ds = FourDS(clf=CMM())
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y, batch_size=1.2)
        al4ds = FourDS(clf=CMM())
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y, batch_size=0)
        al4ds = FourDS(clf=CMM(), lmbda=True)
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)
        al4ds = FourDS(clf=CMM(), lmbda=1.1)
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)
        al4ds = FourDS(clf=self.CMM, random_state=self.random_state)
        query_indices = al4ds.query(X_cand=self.X, X=self.X, y=self.y)
        self.assertEqual((1,), query_indices.shape)
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X,
                                               y=self.y, return_utilities=True)
        self.assertEqual((1,), query_indices.shape)
        self.assertEqual((1, len(self.X)), utilities.shape)
        self.assertEqual(0, np.sum(utilities < 0))
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X,
                                               batch_size=3, y=self.y,
                                               return_utilities=True)
        self.assertEqual((3,), query_indices.shape)
        self.assertEqual((3, len(self.X)), utilities.shape)
        self.assertEqual(3, np.sum(np.isnan(utilities)))
        al4ds = FourDS(clf=self.CMM, random_state=self.random_state)
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X,
                                               batch_size=len(self.X) + 1,
                                               y=self.y, return_utilities=True)
        self.assertEqual((len(self.X),), query_indices.shape)
        self.assertEqual((len(self.X), len(self.X)), utilities.shape)
        self.assertEqual(np.sum(np.arange(0, len(self.X))),
                         np.sum(np.isnan(utilities)))


if __name__ == '__main__':
    unittest.main()
