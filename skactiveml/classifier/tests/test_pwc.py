import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.datasets import make_blobs
from skactiveml.classifier import PWC


class TestPWC(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y_nan = [['nan', 'nan', 'nan'], ['nan', 'nan', 'nan']]
        self.y = [['tokyo', 'nan', 'paris'], ['tokyo', 'nan', 'nan']]
        self.w = [[2, np.nan, 1], [1, 1, 1]]

    def test_init(self):
        pwc = PWC(missing_label=-1)
        self.assertEqual(pwc.missing_label, -1)
        self.assertEqual(pwc.classes, None)
        self.assertEqual(pwc.metric_dict, {})
        self.assertEqual(pwc.random_state, None)
        self.assertEqual(pwc.cost_matrix, None)

    def test_fit(self):
        pwc = PWC(missing_label='nan', metric="Test")
        self.assertRaises(ValueError, pwc.fit, X=self.X, y=self.y)
        pwc = PWC(missing_label='nan', n_neighbors=-1)
        self.assertRaises(ValueError, pwc.fit, X=self.X, y=self.y)
        pwc = PWC(missing_label='nan', n_neighbors=1.0)
        self.assertRaises(TypeError, pwc.fit, X=self.X, y=self.y)
        pwc = PWC(missing_label='nan', metric_dict=['gamma'])
        self.assertRaises(TypeError, pwc.fit, X=self.X, y=self.y)
        pwc = PWC(missing_label='nan', cost_matrix=1 - np.eye(3))
        self.assertRaises(ValueError, pwc.fit, X=self.X, y=self.y)
        pwc = PWC(missing_label=None)
        self.assertRaises(NotFittedError, check_is_fitted, estimator=pwc)
        cost_matrix = 1 - np.eye(2)
        pwc = PWC(classes=['tokyo', 'paris'], cost_matrix=cost_matrix,
                  missing_label='nan')
        np.testing.assert_array_equal(cost_matrix, pwc.cost_matrix)
        pwc = PWC(missing_label='nan')
        self.assertRaises(ValueError, pwc.fit, X=[], y=[])
        self.assertRaises(ValueError,  pwc.fit, X=self.X, y=self.y_nan)
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan')
        pwc.fit(X=self.X, y=self.y_nan)
        self.assertIsNone(pwc.cost_matrix)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix_)
        np.testing.assert_array_equal([[0, 0, 0], [0, 0, 0]], pwc.V_)
        pwc.fit(X=self.X, y=self.y)
        self.assertIsNone(pwc.cost_matrix)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix_)
        np.testing.assert_array_equal([[0, 1, 1], [0, 0, 1]], pwc.V_)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal([[0, 1, 2], [0, 0, 1]], pwc.V_)

    def test_predict_freq(self):
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan',
                  n_neighbors=10, metric='rbf', metric_dict={'gamma': 2})
        self.assertRaises(NotFittedError, pwc.predict_freq, X=self.X)
        pwc.fit(X=self.X, y=self.y_nan)
        F = pwc.predict_freq(X=self.X)
        np.testing.assert_array_equal(np.zeros((len(self.X), 3)), F)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = pwc.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 3]], F)
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan',
                  n_neighbors=1)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = pwc.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 0, 1]], F)
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan',
                  n_neighbors=1, metric='precomputed')
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1, 0, 0]])
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1], [0]])
        F = pwc.predict_freq(X=[[1, 0]])
        np.testing.assert_array_equal([[0, 1, 2]], F)

    def test_predict_proba(self):
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan')
        self.assertRaises(NotFittedError, pwc.predict_proba, X=self.X)
        pwc.fit(X=self.X, y=self.y_nan)
        P = pwc.predict_proba(X=self.X)
        np.testing.assert_array_equal(np.ones((len(self.X), 2)) * 0.5, P)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = pwc.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1 / 4, 3 / 4]], P)

    def test_predict(self):
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan',
                  random_state=0)
        self.assertRaises(NotFittedError, pwc.predict, X=self.X)
        pwc.fit(X=self.X, y=self.y_nan)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'paris'], y)
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan',
                  random_state=1).fit(X=self.X, y=self.y_nan)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan',
                  cost_matrix=[[0, 1, 4], [10, 0, 5], [2, 2, 0]])
        pwc.fit(X=self.X, y=self.y_nan)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan',
                  cost_matrix=[[0, 1], [10, 0]])
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)

    def test_on_data_set(self):
        X, y = make_blobs(n_samples=300, random_state=0)
        pwc = PWC(random_state=0).fit(X, y)
        self.assertTrue(pwc.score(X, y) > 0.5)


if __name__ == '__main__':
    unittest.main()
