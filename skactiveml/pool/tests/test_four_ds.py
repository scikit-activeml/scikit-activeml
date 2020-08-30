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
        mixture_model = BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_distribution')
        mixture_model.fit(self.X)
        self.CMM = CMM(mixture_model=mixture_model)

    def test_fit(self):
        self.assertRaises(TypeError, FourDS, clf=GaussianProcessClassifier())
        self.assertRaises(TypeError, FourDS, clf=CMM(), batch_size=1.2)
        self.assertRaises(ValueError, FourDS, clf=CMM(), batch_size=0)
        self.assertRaises(TypeError, FourDS, clf=CMM(), lmbda=True)
        self.assertRaises(ValueError, FourDS, clf=CMM(), lmbda=1.1)
        al4ds = FourDS(clf=CMM(), batch_size=3, random_state=self.random_state)
        self.assertTrue(isinstance(al4ds.clf, CMM))
        self.assertEqual(al4ds.batch_size, 3)
        self.assertEqual(al4ds.lmbda, 0.1)
        self.assertTrue(isinstance(al4ds.random_state, np.random.RandomState))

    def test_query(self):
        al4ds = FourDS(clf=self.CMM, batch_size=1, random_state=self.random_state)
        query_indices = al4ds.query(X_cand=self.X, X=self.X, y=self.y)
        self.assertEqual((1,), query_indices.shape)
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X, y=self.y, return_utilities=True)
        self.assertEqual((1,), query_indices.shape)
        self.assertEqual((len(self.X),), utilities.shape)
        self.assertEqual(0, np.sum(utilities < 0))
        al4ds = FourDS(clf=self.CMM, batch_size=3, random_state=self.random_state)
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X, y=self.y, return_utilities=True)
        self.assertEqual((3,), query_indices.shape)
        self.assertEqual((len(self.X), 3), utilities.shape)
        self.assertEqual(3, np.sum(np.equal(utilities, -np.inf)))
        al4ds = FourDS(clf=self.CMM, batch_size=len(self.X) + 1, random_state=self.random_state)
        query_indices, utilities = al4ds.query(X_cand=self.X, X=self.X, y=self.y, return_utilities=True)
        self.assertEqual((len(self.X),), query_indices.shape)
        self.assertEqual((len(self.X), len(self.X)), utilities.shape)
        self.assertEqual(np.sum(np.arange(0, len(self.X))), np.sum(np.equal(utilities, -np.inf)))


if __name__ == '__main__':
    unittest.main()
