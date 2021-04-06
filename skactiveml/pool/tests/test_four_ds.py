import numpy as np
import unittest

from skactiveml.pool import FourDS
from skactiveml.classifier import CMM, SklearnClassifier
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

    def test_init_param_clf(self):
        al4ds = FourDS(clf=None)
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)
        clf = SklearnClassifier(GaussianProcessClassifier())
        al4ds = FourDS(clf=clf)
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)

    def test_init_param_lmbda(self):
        al4ds = FourDS(clf=CMM(), lmbda=True)
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)
        al4ds = FourDS(clf=CMM(), lmbda=1.1)
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)

    def test_init_param_random_state(self):
        al4ds = FourDS(clf=CMM(), random_state='tests')
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y)

    def test_query_param_batch_size(self):
        al4ds = FourDS(clf=CMM())
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y, batch_size=1.2)
        al4ds = FourDS(clf=CMM())
        self.assertRaises(ValueError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y, batch_size=0)

    def test_query_param_return_utilities(self):
        al4ds = FourDS(clf=CMM())
        self.assertRaises(TypeError, al4ds.query, X_cand=self.X, X=self.X,
                          y=self.y, return_utilities='test')

    def test_query_param_X_cand(self):
        al4ds = FourDS(clf=CMM())
        self.assertRaises(ValueError, al4ds.query, X_cand=None, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, al4ds.query, X_cand=np.ones(5), X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, al4ds.query, X_cand=np.ones((5, 1)),
                          X=self.X, y=self.y)

    def test_query_param_X(self):
        al4ds = FourDS(clf=CMM())
        self.assertRaises(ValueError, al4ds.query, X=None, X_cand=self.X,
                          y=self.y)
        self.assertRaises(ValueError, al4ds.query, X=np.ones(5), X_cand=self.X,
                          y=self.y)
        self.assertRaises(ValueError, al4ds.query, X=np.ones((5, 1)),
                          X_cand=self.X, y=self.y)

    def test_query_param_y(self):
        al4ds = FourDS(clf=CMM())
        self.assertRaises(ValueError, al4ds.query, X=self.X, X_cand=self.X,
                          y=None)
        self.assertRaises(ValueError, al4ds.query, X=self.X, X_cand=self.X,
                          y=np.zeros((len(self.y), 2)))

    def test_query_param_sample_weight(self):
        al4ds = FourDS(clf=CMM())
        self.assertRaises(ValueError, al4ds.query, X=self.X, X_cand=self.X,
                          y=self.y, sample_weight=np.ones(1))
        self.assertRaises(TypeError, al4ds.query, X=self.X, X_cand=self.X,
                          y=self.y, sample_weight='test')

    def test_query(self):
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
