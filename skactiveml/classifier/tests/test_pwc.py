import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.datasets import load_breast_cancer
from skactiveml.classifier import PWC


class TestPWC(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y = [['tokyo', 'nan', 'paris'], ['tokyo', 'nan', 'nan']]
        self.w = [[2, np.nan, 1], [1, 1, 1]]

    def test_init(self):
        self.assertRaises(ValueError, PWC, metric="Test")
        self.assertRaises(ValueError, PWC, n_neighbors=-1)
        self.assertRaises(TypeError, PWC, n_neighbors=1.0)
        pwc = PWC()
        self.assertRaises(NotFittedError, check_is_fitted, estimator=pwc)
        cost_matrix = 1-np.eye(2)
        pwc = PWC(classes=['tokyo', 'paris'], cost_matrix=cost_matrix, missing_label='nan')
        self.assertIsNotNone(pwc._le)
        np.testing.assert_array_equal(cost_matrix, pwc.cost_matrix)
        self.assertEqual('nan', pwc._le.missing_label)

    def test_fit(self):
        pwc = PWC()
        self.assertRaises(ValueError, pwc.fit, X=[], y=[])
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan').fit(X=[], y=[])
        np.testing.assert_array_equal(1-np.eye(3), pwc.cost_matrix)
        np.testing.assert_array_equal([], pwc.V_)
        pwc.fit(X=self.X, y=self.y)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix)
        np.testing.assert_array_equal([[0, 1, 1], [0, 0, 1]], pwc.V_)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal([[0, 1, 2], [0, 0, 1]], pwc.V_)

    def test_predict_freq(self):
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan', n_neighbors=10)
        F = pwc.predict_freq(X=self.X)
        np.testing.assert_array_equal(np.zeros((len(self.X), 3)), F)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = pwc.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 3]], F)
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan', n_neighbors=1)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = pwc.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 0, 1]], F)
        pwc = PWC(classes=['tokyo', 'paris', 'new york'], missing_label='nan', n_neighbors=1, metric='precomputed')
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1, 0, 0]])
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1], [0]])
        F = pwc.predict_freq(X=[[1, 0]])
        np.testing.assert_array_equal([[0, 1, 2]], F)

    def test_predict_proba(self):
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan')
        P = pwc.predict_proba(X=self.X)
        np.testing.assert_array_equal(np.ones((len(self.X), 2))*0.5, P)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = pwc.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1/4, 3/4]], P)

    def test_predict(self):
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan', random_state=0)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'paris'], y)
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan', random_state=1).fit(X=[], y=[])
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['tokyo', 'tokyo'], y)
        pwc = PWC(classes=['tokyo', 'paris'], missing_label='nan', cost_matrix=[[0, 10], [1, 0]])
        pwc.fit(X=[], y=[])
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)
        pwc = PWC(missing_label='nan', cost_matrix=[[0, 10], [1, 0]])
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(['paris', 'paris'], y)
        X, y = load_breast_cancer(return_X_y=True)
        pwc = PWC(random_state=0).fit(X, y)
        self.assertTrue(pwc.score(X, y) > 0.5)


if __name__ == '__main__':
    unittest.main()
