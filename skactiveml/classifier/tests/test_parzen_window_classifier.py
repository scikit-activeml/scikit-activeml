import unittest

import numpy as np
from sklearn.utils.validation import NotFittedError

from skactiveml.classifier import ParzenWindowClassifier


class TestParzenWindowClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((3, 1))
        self.y_nan = ["nan", "nan", "nan"]
        self.y = ["tokyo", "nan", "paris"]
        self.w = [2, np.nan, 1]

    def test_init_param_metric_dict(self):
        pwc = ParzenWindowClassifier(missing_label=-1)
        self.assertEqual(pwc.metric_dict, None)
        pwc = ParzenWindowClassifier(missing_label="nan", metric_dict="Test")
        self.assertRaises(TypeError, pwc.fit, X=self.X, y=self.y)
        pwc = ParzenWindowClassifier(
            missing_label="nan", metric_dict=["gamma"]
        )
        self.assertRaises(TypeError, pwc.fit, X=self.X, y=self.y)

    def test_init_param_metric(self):
        pwc = ParzenWindowClassifier()
        self.assertEqual(pwc.metric, "rbf")
        pwc = ParzenWindowClassifier(metric="Test")
        self.assertEqual(pwc.metric, "Test")
        pwc = ParzenWindowClassifier(missing_label="nan", metric="Test")
        self.assertRaises(ValueError, pwc.fit, X=self.X, y=self.y)

    def test_init_param_n_neighbors(self):
        pwc = ParzenWindowClassifier()
        self.assertTrue(pwc.n_neighbors is None)
        pwc = ParzenWindowClassifier(n_neighbors=1)
        self.assertEqual(pwc.n_neighbors, 1)
        pwc = ParzenWindowClassifier(missing_label="nan", n_neighbors=0)
        self.assertRaises(ValueError, pwc.fit, X=self.X, y=self.y)
        pwc = ParzenWindowClassifier(missing_label="nan", n_neighbors=-1)
        self.assertRaises(ValueError, pwc.fit, X=self.X, y=self.y)
        pwc = ParzenWindowClassifier(missing_label="nan", n_neighbors=1.5)
        self.assertRaises(TypeError, pwc.fit, X=self.X, y=self.y)

    def test_fit(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"], missing_label="nan"
        )
        pwc.fit(X=self.X, y=self.y_nan)
        self.assertIsNone(pwc.cost_matrix)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix_)
        np.testing.assert_array_equal(np.zeros((3, 3)), pwc.V_)
        pwc.fit(X=self.X, y=self.y)
        self.assertIsNone(pwc.cost_matrix)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix_)
        np.testing.assert_array_equal(
            [[0, 0, 1], [0, 0, 0], [0, 1, 0]], pwc.V_
        )
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal(
            [[0, 0, 2], [0, 0, 0], [0, 1, 0]], pwc.V_
        )

    def test_predict_freq(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=10,
            metric="rbf",
            metric_dict={"gamma": 2},
        )
        self.assertRaises(NotFittedError, pwc.predict_freq, X=self.X)
        pwc.fit(X=self.X, y=self.y_nan)
        F = pwc.predict_freq(X=self.X)
        np.testing.assert_array_equal(np.zeros((len(self.X), 3)), F)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = pwc.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 2]], F)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
        )
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = pwc.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 0]], F)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
            metric="precomputed",
        )
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1, 0]])
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1], [0]])
        F = pwc.predict_freq(X=[[1, 0, 0]])
        np.testing.assert_array_equal([[0, 0, 2]], F)
        rbf_kernel = lambda x, y, gamma: np.exp(-gamma * np.sum((x - y) ** 2))
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"],
            missing_label="nan",
            random_state=0,
            metric=rbf_kernel,
            metric_dict={"gamma": 2},
        )
        F_call = pwc.fit(X=self.X, y=self.y).predict_freq(np.ones_like(self.X))
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"],
            missing_label="nan",
            metric="rbf",
            metric_dict={"gamma": 2},
            random_state=0,
        )
        F_rbf = pwc.fit(X=self.X, y=self.y).predict_freq(np.ones_like(self.X))
        np.testing.assert_array_equal(F_call, F_rbf)

    def test_predict_proba(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"], missing_label="nan"
        )
        self.assertRaises(NotFittedError, pwc.predict_proba, X=self.X)
        pwc.fit(X=self.X, y=self.y_nan)
        P = pwc.predict_proba(X=self.X)
        np.testing.assert_array_equal(np.ones((len(self.X), 2)) * 0.5, P)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = pwc.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1 / 3, 2 / 3]], P)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
            metric="precomputed",
            class_prior=1,
        )
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = pwc.predict_proba(X=[[1, 0, 0]])
        np.testing.assert_array_equal([[1 / 5, 1 / 5, 3 / 5]], P)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
            metric="precomputed",
            class_prior=[0, 0, 1],
        )
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = pwc.predict_proba(X=[[1, 0, 0]])
        np.testing.assert_array_equal([[0, 0, 1]], P)

    def test_predict(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"], missing_label="nan", random_state=0
        )
        self.assertRaises(NotFittedError, pwc.predict, X=self.X)
        pwc.fit(X=self.X, y=self.y_nan)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(["tokyo", "paris", "tokyo"], y)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"], missing_label="nan", random_state=1
        )
        pwc.fit(X=self.X, y=self.y_nan)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(["tokyo", "tokyo", "paris"], y)
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(["tokyo", "tokyo", "tokyo"], y)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            cost_matrix=[[0, 1, 4], [10, 0, 5], [2, 2, 0]],
        )
        pwc.fit(X=self.X, y=self.y_nan)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"],
            missing_label="nan",
            cost_matrix=[[0, 1], [10, 0]],
        )
        pwc.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = pwc.predict(self.X)
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
