import unittest
import warnings

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import NotFittedError, check_is_fitted

from skactiveml.classifier import (
    SklearnClassifier,
    SlidingWindowClassifier,
    ParzenWindowClassifier,
)


class TestSklearnClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((4, 1))
        self.y1 = ["tokyo", "paris", "nan", "tokyo"]
        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    def test_init_param_estimator(self):
        clf = SklearnClassifier(estimator="Test")
        self.assertEqual(clf.estimator, "Test")
        clf = SklearnClassifier(estimator="Test")
        self.assertEqual(clf.estimator, "Test")
        clf = SklearnClassifier(
            missing_label="nan", estimator=GaussianProcessRegressor()
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_fit(self):
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            missing_label="nan",
            classes=["tokyo", "paris"],
            random_state=0,
        )
        np.testing.assert_array_equal(["tokyo", "paris"], clf.classes)
        self.assertEqual(clf.kernel, clf.estimator.kernel)
        self.assertFalse(hasattr(clf, "kernel_"))
        clf = SklearnClassifier(
            estimator=Perceptron(),
            missing_label="nan",
            cost_matrix=1 - np.eye(2),
            classes=["tokyo", "paris"],
            random_state=0,
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)
        clf = SklearnClassifier(estimator=GaussianProcessClassifier())
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf.fit(self.X, self.y1, sample_weight=np.ones_like(self.y1))
        self.assertTrue(clf.is_fitted_)
        clf.fit(self.X, self.y1)
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "kernel_"))
        np.testing.assert_array_equal(
            clf.classes_, ["new york", "paris", "tokyo"]
        )
        self.assertEqual(clf.missing_label, "nan")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(self.X, self.y2)
            self.assertEqual(len(w), 1)
        self.assertFalse(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        self.assertFalse(hasattr(clf, "partial_fit"))

        X = [[1], [0]]
        y_true = [1, 0]
        clf = SklearnClassifier(GaussianProcessClassifier(), classes=[0, 1])
        ensemble = SklearnClassifier(BaggingClassifier(clf), classes=[0, 1])
        ensemble.fit(X, y_true)
        self.assertTrue(ensemble.is_fitted_, True)

    def test_partial_fit(self):
        clf = SklearnClassifier(
            estimator=GaussianNB(),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf.partial_fit(self.X, self.y1)
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "class_count_"))
        np.testing.assert_array_equal(
            clf.classes_, ["new york", "paris", "tokyo"]
        )
        self.assertEqual(clf.missing_label, "nan")
        clf.partial_fit(self.X, self.y2, sample_weight=np.ones_like(self.y2))
        self.assertTrue(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        self.assertTrue(hasattr(clf, "partial_fit"))
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertFalse(hasattr(clf, "partial_fit"))

    def test_predict_proba(self):
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(), missing_label="nan"
        )
        self.assertRaises(NotFittedError, clf.predict_proba, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        P = clf.predict_proba(X=self.X)
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        P_exp = est.predict_proba(X=self.X)
        np.testing.assert_array_equal(P_exp, P)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.X, y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P = clf.predict_proba(X=self.X)
            self.assertEqual(len(w), 1)
        P_exp = np.ones((len(self.X), 1))
        np.testing.assert_array_equal(P_exp, P)
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            classes=["ny", "paris", "tokyo"],
            missing_label="nan",
        )
        clf.fit(X=self.X, y=self.y_nan)
        P = clf.predict_proba(X=self.X)
        P_exp = np.ones((len(self.X), 3)) / 3
        np.testing.assert_array_equal(P_exp, P)
        clf.fit(X=self.X, y=self.y1)
        P = clf.predict_proba(X=self.X)
        P_exp = np.zeros((len(self.X), 3))
        P_exp[:, 1:] = est.predict_proba(X=self.X)
        np.testing.assert_array_equal(P_exp, P)
        clf = SklearnClassifier(
            estimator=Perceptron(),
            classes=["ny", "paris", "tokyo"],
            missing_label="nan",
        )
        self.assertFalse(hasattr(clf, "predict_proba"))

    def test_predict(self):
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(), missing_label="nan"
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        y = clf.predict(X=self.X)
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        y_exp = est.predict(X=self.X)
        np.testing.assert_array_equal(y, y_exp)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.X, y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = clf.predict(X=self.X)
            self.assertEqual(len(w), 1)
        y_exp = ["tokyo"] * len(self.X)
        np.testing.assert_array_equal(y_exp, y)


class TestSlidingWindowClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((4, 1))
        self.y1 = ["tokyo", "paris", "nan", "tokyo"]
        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y3 = [0, 1, 0, 0]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    def test_init_param_estimator(self):
        est = GaussianNB()
        clf = SlidingWindowClassifier(estimator=est, missing_label="nan")
        self.assertEqual(clf.estimator, est)
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_window_size(self):
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(), window_size="Test"
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(), window_size=-1
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_only_labeled(self):
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(), only_labeled="Test"
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(), only_labeled=0
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_ignore_estimator_partial_fit(self):
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(),
            ignore_estimator_partial_fit="Test",
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(), ignore_estimator_partial_fit=0
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_fit(self):
        # check if clf is correctly initialized
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(),
                missing_label="nan",
                classes=["tokyo", "paris"],
            ),
            missing_label="nan",
            classes=["tokyo", "paris"],
            random_state=0,
        )
        np.testing.assert_array_equal(["tokyo", "paris"], clf.classes)
        self.assertEqual(clf.estimator.kernel, clf.estimator.estimator.kernel)
        self.assertFalse(hasattr(clf, "kernel_"))

        # check cost matrix
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(Perceptron(), missing_label="nan"),
            missing_label="nan",
            cost_matrix=1 - np.eye(2),
            classes=["tokyo", "paris"],
            random_state=0,
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)

        clf = SlidingWindowClassifier(estimator=GaussianProcessClassifier())
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)

        # check if classifier is correctly fitted
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(),
                classes=["new york", "paris", "tokyo"],
                missing_label="nan",
            ),
            classes=["new york", "paris", "tokyo"],
            missing_label="nan",
            only_labeled=True,
        )
        clf.fit(self.X, self.y1)
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "kernel_"))
        np.testing.assert_array_equal(
            clf.estimator_.classes_, ["new york", "paris", "tokyo"]
        )
        self.assertEqual(clf.missing_label, "nan")
        # test if warnings are correctly handeled
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(self.X, self.y2)
            self.assertEqual(len(w), 1)
        self.assertFalse(clf.is_fitted_)
        self.assertFalse(clf.estimator_.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        # fit clf with no prior classes and no labels
        clf = SlidingWindowClassifier(
            SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y_nan)
        # fit clf with correct data and sample_weight
        clf = SlidingWindowClassifier(
            SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        clf.fit(self.X, self.y1, sample_weight=np.ones(len(self.y1)))

        X = [[1], [0]]
        y_true = [1, 0]
        clf = SlidingWindowClassifier(
            SklearnClassifier(GaussianProcessClassifier()), classes=[0, 1]
        )
        ensemble = SlidingWindowClassifier(
            SklearnClassifier(BaggingClassifier(clf)), classes=[0, 1]
        )
        ensemble.fit(X, y_true)
        self.assertTrue(ensemble.is_fitted_, True)

    def test_partial_fit(self):
        # check if clf is correctly initialized
        clf = SlidingWindowClassifier(
            SklearnClassifier(estimator=GaussianNB(), missing_label="nan"),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf.partial_fit(self.X, self.y1)
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "class_count_"))
        # check if cost matrix is equal
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                BaggingClassifier(),
                missing_label="nan",
                classes=["tokyo", "paris", "new york"],
                cost_matrix=[[1, 2, 1], [2, 1, 1], [2, 1, 3]],
            ),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labeled=True,
            window_size=5,
            cost_matrix=[[1, 1, 1], [2, 1, 1], [2, 1, 3]],
        )
        # test if clf functions complete data and only_labeled=True
        self.assertRaises(ValueError, clf.partial_fit, X=self.X, y=self.y1)
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                BaggingClassifier(),
                missing_label="nan",
                classes=["tokyo", "paris", "new york"],
            ),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labeled=True,
            window_size=5,
        )
        clf.partial_fit(self.X, self.y1, sample_weight=np.ones_like(self.y1))
        self.assertTrue(clf.is_fitted_)

        clf = SlidingWindowClassifier(estimator=GaussianProcessClassifier())
        self.assertRaises(TypeError, clf.partial_fit, self.X, self.y1)

        # test if clf functions with complete data
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianNB(),
                classes=["tokyo", "paris", "new york"],
                missing_label="nan",
            ),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labeled=False,
            window_size=5,
        )
        self.assertEqual(clf.missing_label, "nan")
        clf.partial_fit(
            self.X, self.y_nan, sample_weight=np.ones_like(self.y_nan)
        )
        clf.partial_fit(self.X, self.y2, sample_weight=np.ones_like(self.y2))
        self.assertTrue(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        clf.partial_fit(self.X, self.y2, sample_weight=np.ones_like(self.y2))
        self.assertEqual(len(clf.X_train_), 5)
        clf.partial_fit(
            self.X, self.y_nan, sample_weight=np.ones_like(self.y2)
        )
        # test clf with classes and empty data
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianNB(),
                classes=["tokyo", "paris", "new york"],
                missing_label="nan",
            ),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labeled=False,
            window_size=5,
            ignore_estimator_partial_fit=True,
        )
        self.assertEqual(clf.missing_label, "nan")
        clf.partial_fit(
            self.X, self.y_nan, sample_weight=np.ones_like(self.y2)
        )
        clf.partial_fit(self.X, self.y2, sample_weight=np.ones_like(self.y2))
        self.assertTrue(clf.is_fitted_)

    def test_predict_proba(self):
        clf = SlidingWindowClassifier(
            SklearnClassifier(
                estimator=GaussianProcessClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, clf.predict_proba, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        P = clf.predict_proba(X=self.X)
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        P_exp = est.predict_proba(X=self.X)
        np.testing.assert_array_equal(P_exp, P)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.X, y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P = clf.predict_proba(X=self.X)
            self.assertEqual(len(w), 1)
        P_exp = np.ones((len(self.X), 1))
        np.testing.assert_array_equal(P_exp, P)
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(),
                missing_label="nan",
                classes=["ny", "paris", "tokyo"],
            ),
            classes=["ny", "paris", "tokyo"],
            missing_label="nan",
        )
        clf.fit(X=self.X, y=self.y_nan)
        P = clf.predict_proba(X=self.X)
        P_exp = np.ones((len(self.X), 3)) / 3
        np.testing.assert_array_equal(P_exp, P)
        clf.fit(X=self.X, y=self.y1)
        P = clf.predict_proba(X=self.X)
        P_exp = np.zeros((len(self.X), 3))
        P_exp[:, 1:] = est.predict_proba(X=self.X)
        np.testing.assert_array_equal(P_exp, P)

    def test_predict(self):
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        y = clf.predict(X=self.X)
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        y_exp = est.predict(X=self.X)
        # Predicts wrong classes (numbers instead of strings)
        np.testing.assert_array_equal(y, y_exp)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.X, y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = clf.predict(X=self.X)
            self.assertEqual(len(w), 1)
        y_exp = ["tokyo"] * len(self.X)
        np.testing.assert_array_equal(y_exp, y)

    def test_predict_freq(self):
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                ParzenWindowClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, clf.predict_freq, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        freq = clf.predict_freq(X=self.X)

        self.assertEqual(len(np.unique(freq)), 2)
        est = ParzenWindowClassifier(missing_label="nan").fit(
            X=self.X, y=self.y1
        )
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                ParzenWindowClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )

        clf.fit(X=self.X, y=self.y1)
        freq = clf.predict_freq(X=self.X)
        est.fit(X=self.X, y=self.y1)
        freq_est = est.predict_freq(X=self.X)
        np.testing.assert_array_equal(freq, freq_est)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
