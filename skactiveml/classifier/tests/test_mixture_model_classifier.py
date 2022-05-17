import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils.validation import NotFittedError, check_is_fitted

from skactiveml.classifier import MixtureModelClassifier


class TestMixtureModelClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((3, 1))
        self.y = ["tokyo", "nan", "paris"]
        self.y_nan = ["nan", "nan", "nan"]
        self.w = [2, np.nan, 1]

    def test_init_param_mixture_model(self):
        cmm = MixtureModelClassifier(missing_label=-1)
        self.assertEqual(cmm.mixture_model, None)
        cmm = MixtureModelClassifier(missing_label="nan", mixture_model="Test")
        self.assertRaises(TypeError, cmm.fit, X=self.X, y=self.y)

    def test_init_param_weight_mode(self):
        cmm = MixtureModelClassifier(missing_label=-1)
        self.assertEqual(cmm.weight_mode, "responsibilities")
        cmm = MixtureModelClassifier(missing_label="nan", weight_mode="Test")
        self.assertRaises(ValueError, cmm.fit, X=self.X, y=self.y)

    def test_fit(self):
        mixture = GaussianMixture(random_state=0, n_components=4)
        cmm = MixtureModelClassifier(
            missing_label="nan",
            mixture_model=mixture,
            classes=[1, 2],
            cost_matrix=1 - np.eye(3),
        )
        self.assertRaises(TypeError, cmm.fit, X=self.X, y=self.y)
        cmm = MixtureModelClassifier(
            missing_label="nan",
            mixture_model=mixture,
            cost_matrix=1 - np.eye(3),
        )
        self.assertRaises(ValueError, cmm.fit, X=self.X, y=self.y)
        cmm = MixtureModelClassifier(missing_label=None, random_state=0)
        self.assertRaises(NotFittedError, check_is_fitted, estimator=cmm)
        cost_matrix = 1 - np.eye(2)
        cmm = MixtureModelClassifier(
            classes=["tokyo", "paris"],
            cost_matrix=cost_matrix,
            missing_label="nan",
        )
        np.testing.assert_array_equal(cost_matrix, cmm.cost_matrix)
        self.assertEqual("nan", cmm.missing_label)
        self.assertEqual(cmm.mixture_model, None)
        np.testing.assert_array_equal(["tokyo", "paris"], cmm.classes)
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertEqual(None, cmm.cost_matrix)
        self.assertFalse(hasattr(cmm, "F_components_"))
        self.assertFalse(hasattr(cmm, "_refit"))
        self.assertFalse(hasattr(cmm, "classes_"))
        cmm.fit(X=self.X, y=self.y)
        self.assertTrue(hasattr(cmm, "mixture_model_"))
        np.testing.assert_array_equal(
            cmm.classes_, ["new york", "paris", "tokyo"]
        )
        np.testing.assert_array_equal(1 - np.eye(3), cmm.cost_matrix_)
        np.testing.assert_array_equal([[0, 1, 1]], cmm.F_components_)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal([[0, 1, 2]], cmm.F_components_)

    def test_predict_freq(self):
        mixture = BayesianGaussianMixture(n_components=1)
        mixture.fit(X=self.X, y=self.y)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, cmm.predict_freq, X=self.X)
        cmm.fit(X=self.X, y=self.y_nan)
        F = cmm.predict_freq(X=self.X)
        np.testing.assert_array_equal(np.zeros((len(self.X), 3)), F)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        F = cmm.predict_freq(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1, 2]], F)
        X, y = make_blobs(n_samples=200, centers=2)
        y_nan = np.full_like(y, np.nan, dtype=float)
        mixture = BayesianGaussianMixture(n_components=5)
        cmm = MixtureModelClassifier(
            mixture_model=mixture, classes=[0, 1], weight_mode="similarities"
        )
        self.assertRaises(NotFittedError, cmm.predict_freq, X=self.X)
        cmm.fit(X=X, y=y_nan)
        F = cmm.predict_freq(X=X)
        np.testing.assert_array_equal(F.shape, [200, 2])
        self.assertEqual(F.sum(), 0)
        cmm.fit(X=X, y=y)
        F = cmm.predict_freq(X=X)
        self.assertTrue(F.sum() > 0)

    def test_predict_proba(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(X=self.X)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, cmm.predict_proba, X=self.X)
        cmm.fit(X=self.X, y=self.y_nan)
        P = cmm.predict_proba(X=self.X)
        np.testing.assert_array_equal(np.ones((len(self.X), 2)) * 0.5, P)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = cmm.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1 / 3, 2 / 3]], P)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            missing_label="nan",
            classes=["tokyo", "paris", "new york"],
            class_prior=1,
        )
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = cmm.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[1 / 6, 2 / 6, 3 / 6]], P)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            missing_label="nan",
            classes=["tokyo", "paris", "new york"],
            class_prior=[0, 0, 1],
        )
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        P = cmm.predict_proba(X=[self.X[0]])
        np.testing.assert_array_equal([[0, 1 / 4, 3 / 4]], P)

    def test_predict(self):
        mixture = BayesianGaussianMixture(n_components=1, random_state=0)
        mixture.fit(X=self.X)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            random_state=0,
        )
        self.assertRaises(NotFittedError, cmm.predict, X=self.X)
        cmm.fit(X=self.X, y=self.y_nan)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(["paris", "tokyo", "tokyo"], y)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris"],
            missing_label="nan",
            random_state=1,
        )
        cmm.fit(X=self.X, y=self.y_nan)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(["tokyo", "tokyo", "paris"], y)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(["tokyo", "tokyo", "tokyo"], y)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris"],
            missing_label="nan",
            cost_matrix=[[0, 1], [10, 0]],
        )
        cmm.fit(X=self.X, y=self.y)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
        cmm.fit(X=self.X, y=self.y, sample_weight=self.w)
        y = cmm.predict(self.X)
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
