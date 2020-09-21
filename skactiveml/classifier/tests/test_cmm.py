import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.datasets import make_blobs
from skactiveml.classifier import CMM


class TestCMM(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y = [['tokyo', 'nan', 'paris'], ['tokyo', 'nan', 'nan']]
        self.y_nan = [['nan', 'nan', 'nan'], ['nan', 'nan', 'nan']]
        self.w = [[2, np.nan, 1], [1, 1, 1]]

    def test_init(self):
        self.assertRaises(TypeError, CMM, mixture_model="Test")
        mixture = GaussianMixture(random_state=0, n_components=4)
        self.assertRaises(ValueError, CMM, mixture_model=mixture,
                          classes=[1, 2], cost_matrix=1 - np.eye(3))
        self.assertRaises(ValueError, CMM, mixture_model=mixture,
                          cost_matrix=1 - np.eye(3))
        cmm = CMM(mixture_model=mixture, missing_label=None, random_state=0)
        self.assertRaises(NotFittedError, check_is_fitted, estimator=cmm)
        cost_matrix = 1 - np.eye(2)
        cmm = CMM(classes=['tokyo', 'paris'],
                  cost_matrix=cost_matrix, missing_label='nan')
        np.testing.assert_array_equal(cost_matrix, cmm.cost_matrix)
        self.assertEqual('nan', cmm.missing_label)
        self.assertEqual(cmm.mixture_model.n_components, 10)
        np.testing.assert_array_equal(['tokyo', 'paris'], cmm.classes)

    def test_fit(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X)
        cmm = CMM(mixture_model=mixture)
        self.assertRaises(ValueError, cmm.fit, X=[], y=[])
        cmm = CMM(mixture_model=mixture,
                  classes=['tokyo', 'paris', 'new york'], missing_label='nan')
        self.assertEqual(None, cmm.cost_matrix)
        self.assertFalse(hasattr(cmm, 'F_components_'))
        self.assertFalse(hasattr(cmm, '_refit'))
        self.assertFalse(hasattr(cmm, 'classes_'))
        cmm.fit(X=self.X, y=self.y)
        self.assertTrue(hasattr(cmm, 'mixture_model_'))
        np.testing.assert_array_equal(cmm.classes_,
                                      ['new york', 'paris', 'tokyo'])
        np.testing.assert_array_equal(1 - np.eye(3), cmm.cost_matrix_)
        np.testing.assert_array_equal([[0, 1, 2]], cmm.F_components_)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal([[0, 1, 3]], cmm.F_components_)

    def test_predict_freq(self):
        mixture = BayesianGaussianMixture(n_components=1)
        mixture.fit(X=self.X, y=self.y)
        cmm = CMM(mixture_model=mixture,
                  classes=['tokyo', 'paris', 'new york'], missing_label='nan')
        self.assertRaises(NotFittedError, cmm.predict_freq, X=self.X)
        cmm.fit(X=self.X, y=self.y_nan)
        F = cmm.predict_freq(X=self.X)
        np.testing.assert_array_equal(np.zeros((len(self.X), 3)), F)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = cmm.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 3]], F)

    def test_predict_proba(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'],
                  missing_label='nan')
        self.assertRaises(NotFittedError, cmm.predict_proba, X=self.X)
        cmm.fit(X=self.X, y=self.y_nan)
        P = cmm.predict_proba(X=self.X)
        np.testing.assert_array_equal(np.ones((len(self.X), 2)) * 0.5, P)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = cmm.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1 / 4, 3 / 4]], P)

    def test_predict(self):
        mixture = BayesianGaussianMixture(n_components=1, random_state=0)
        mixture.fit(X=self.X)
        cmm = CMM(mixture_model=mixture,
                  classes=['tokyo', 'paris', 'new york'], missing_label='nan',
                  random_state=0)
        self.assertRaises(NotFittedError, cmm.predict, X=self.X)
        cmm.fit(X=self.X, y=self.y_nan)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['paris', 'tokyo'], y)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'],
                  missing_label='nan', random_state=1)
        cmm.fit(X=self.X, y=self.y_nan)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        cmm = CMM(mixture_model=mixture, classes=['tokyo', 'paris'],
                  missing_label='nan', cost_matrix=[[0, 1], [10, 0]])
        cmm.fit(X=self.X, y=self.y)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)

    def test_on_data_set(self):
        X, y = make_blobs(n_samples=300, random_state=0)
        mixture_model = BayesianGaussianMixture(n_components=10)
        pwc = CMM(mixture_model=mixture_model, random_state=0).fit(X, y)
        self.assertTrue(pwc.score(X, y) > 0.5)


if __name__ == '__main__':
    unittest.main()
