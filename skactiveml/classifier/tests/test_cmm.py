import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.datasets import load_breast_cancer
from skactiveml.classifier import CMM


class TestCMM(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y = [['tokyo', 'nan', 'paris'], ['tokyo', 'nan', 'nan']]
        self.w = [[2, np.nan, 1], [1, 1, 1]]

    def test_init(self):
        self.assertRaises(TypeError, CMM, mixture_model="Test")
        mixture = GaussianMixture()
        cmm = CMM(mixture_model=GaussianMixture())
        self.assertTrue(cmm._refit)
        mixture.fit(X=self.X)
        self.assertRaises(ValueError, CMM, mixture_model=mixture, classes=[1, 2], cost_matrix=1-np.eye(3))
        cmm = CMM(mixture_model=mixture)
        self.assertRaises(NotFittedError, check_is_fitted, estimator=cmm)
        cost_matrix = 1-np.eye(2)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'], cost_matrix=cost_matrix, missing_label='nan')
        self.assertIsNotNone(cmm._le)
        np.testing.assert_array_equal(cost_matrix, cmm.cost_matrix)
        self.assertEqual('nan', cmm._le.missing_label)
        self.assertFalse(cmm._refit)

    def test_fit(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X)
        cmm = CMM(mixture_model=mixture)
        self.assertRaises(ValueError, cmm.fit, X=[], y=[])
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris', 'new york'], missing_label='nan').fit(X=[], y=[])
        np.testing.assert_array_equal(1-np.eye(3), cmm.cost_matrix)
        np.testing.assert_array_equal(np.zeros((1, 3)), cmm.F_components_)
        cmm.fit(X=self.X, y=self.y)
        np.testing.assert_array_equal(1 - np.eye(3), cmm.cost_matrix)
        np.testing.assert_array_equal([[0, 1, 2]], cmm.F_components_)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal([[0, 1, 3]], cmm.F_components_)

    def test_predict_freq(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X, y=self.y)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris', 'new york'], missing_label='nan')
        F = cmm.predict_freq(X=self.X)
        np.testing.assert_array_equal(np.zeros((len(self.X), 3)), F)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = cmm.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 3]], F)

    def test_predict_proba(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'], missing_label='nan')
        P = cmm.predict_proba(X=self.X)
        np.testing.assert_array_equal(np.ones((len(self.X), 2))*0.5, P)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = cmm.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1/4, 3/4]], P)

    def test_predict(self):
        mixture = BayesianGaussianMixture(n_components=1, random_state=0).fit(X=self.X)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris', 'new york'], missing_label='nan', random_state=0)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['paris', 'tokyo'], y)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'], missing_label='nan', random_state=1)
        y = cmm.fit(X=[], y=[]).predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'], missing_label='nan', cost_matrix=[[0, 10], [1, 0]])
        y = cmm.fit(X=[], y=[]).predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)
        X, y = load_breast_cancer(return_X_y=True)
        cmm = CMM(random_state=0).fit(X, y)
        self.assertEqual(cmm.mixture_model.n_components, 10)
        self.assertTrue(cmm._refit)
        self.assertTrue(cmm.score(X, y) > 0.5)


if __name__ == '__main__':
    unittest.main()
