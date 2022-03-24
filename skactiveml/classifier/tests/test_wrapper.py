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
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import NotFittedError, check_is_fitted

from skactiveml.classifier import (
    SklearnClassifier,
    KernelFrequencyClassifier,
    PWC,
    SubSampleEstimator,
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


class TestKernelFrequencyClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((4, 1))
        self.y1 = ["tokyo", "paris", "nan", "tokyo"]
        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    def test_init_param_estimator(self):
        clf = KernelFrequencyClassifier(estimator="Test")
        self.assertEqual(clf.estimator, "Test")
        clf = KernelFrequencyClassifier(
            missing_label="nan", estimator=SklearnClassifier(GaussianProcessRegressor())
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_frequency_estimator(self):
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(Perceptron()), frequency_estimator="Test"
        )
        self.assertEqual(clf.frequency_estimator, "Test")
        clf = KernelFrequencyClassifier(
            missing_label="nan",
            estimator=SklearnClassifier(GaussianProcessRegressor()),
            frequency_estimator=KernelDensity(),
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_frequency_max_fit_len(self):
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(Perceptron()), frequency_max_fit_len="Test"
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(Perceptron()), frequency_max_fit_len=0
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)

    def test_fit(self):
        # self.assertEqual(clf.kernel, clf.estimator.kernel)
        # self.assertFalse(hasattr(clf, "kernel_"))
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(Perceptron(), missing_label="nan"),
            missing_label="nan",
            cost_matrix=1 - np.eye(2),
            classes=["tokyo", "paris"],
            random_state=0,
        )
        np.testing.assert_array_equal(["tokyo", "paris"], clf.classes)
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)
        clf = KernelFrequencyClassifier(estimator=GaussianProcessClassifier())
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(GaussianProcessClassifier(), classes=["tokyo", "paris", "new york"], missing_label="nan"),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
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
            self.assertEqual(len(w), 2)
        self.assertFalse(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        self.assertFalse(hasattr(clf, "partial_fit"))

        X = [[1], [0]]
        y_true = [1, 0]
        clf = KernelFrequencyClassifier(
            SklearnClassifier(GaussianProcessClassifier(), classes=[0, 1])
        )
        ensemble = KernelFrequencyClassifier(
            SklearnClassifier(BaggingClassifier(clf), classes=[0, 1])
        )
        ensemble.fit(X, y_true)
        self.assertTrue(ensemble.is_fitted_, True)

    def test_partial_fit(self):
        clf = KernelFrequencyClassifier(
            SklearnClassifier(estimator=GaussianNB(), missing_label="nan"),
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
        clf = KernelFrequencyClassifier(
            SklearnClassifier(estimator=GaussianNB(), missing_label="nan"),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertEqual(clf.missing_label, "nan")
        clf.partial_fit(self.X, self.y2, sample_weight=np.ones_like(self.y2))
        self.assertTrue(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        # is allways true since it simmulates partial_fit
        self.assertTrue(hasattr(clf, "partial_fit"))

    def test_predict_proba(self):
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(GaussianProcessClassifier(), missing_label="nan"),
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
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(GaussianProcessClassifier(), classes=["ny", "paris", "tokyo"],  missing_label="nan"),
            classes=["ny", "paris", "tokyo"],
            missing_label="nan",
        )
        # Fehlermeldung kommt wie eigentlich erwartet
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
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(GaussianProcessClassifier()),
            missing_label="nan",
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
        # selbe problem wie bei subsam.. ist int liste und nicht string
        np.testing.assert_array_equal(y_exp, y)

    def test_predict_freq(self):
        clf = KernelFrequencyClassifier(
            estimator=SklearnClassifier(GaussianProcessClassifier(), missing_label="nan"),
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, clf.predict_freq, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        freq = clf.predict_freq(X=self.X)
        est = PWC(missing_label="nan").fit(X=self.X, y=self.y1)
        # Weiß ich nicht genau ob test geändert werden muss oder freq falsch ist
        self.assertEqual(len(np.unique(freq)), 2)
        np.testing.assert_array_equal(clf.classes_, est.classes_)


class TestSubSampleEstimator(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((4, 1))
        self.y1 = ["tokyo", "paris", "nan", "tokyo"]
        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    def test_init_param_estimator(self):
        clf = SubSampleEstimator(estimator="Test")
        self.assertEqual(clf.estimator, "Test")
        clf = SubSampleEstimator(
            missing_label="nan", estimator=GaussianProcessRegressor()
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_max_fit_len(self):
        clf = SubSampleEstimator(estimator=PWC(), max_fit_len="Test")
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = SubSampleEstimator(estimator=PWC(), max_fit_len=-1)
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_handle_window(self):
        clf = SubSampleEstimator(estimator=PWC(), handle_window=None)
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = SubSampleEstimator(estimator=PWC(), handle_window=0)
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_init_param_only_labled(self):
        clf = SubSampleEstimator(estimator=PWC(), only_labled="Test")
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)
        clf = SubSampleEstimator(estimator=PWC(), only_labled=0)
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y1)

    def test_fit(self):
        clf = SubSampleEstimator(
            estimator=SklearnClassifier(GaussianProcessClassifier(), missing_label="nan",
            classes=["tokyo", "paris"]),
            missing_label="nan",
            classes=["tokyo", "paris"],
            random_state=0,
        )
        np.testing.assert_array_equal(["tokyo", "paris"], clf.classes)
        self.assertEqual(clf.estimator.kernel, clf.estimator.estimator.kernel)
        self.assertFalse(hasattr(clf, "kernel_"))
        clf = SubSampleEstimator(
            estimator=SklearnClassifier(Perceptron()),
            missing_label="nan",
            cost_matrix=1 - np.eye(2),
            classes=["tokyo", "paris"],
            random_state=0,
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y1)
        clf = SubSampleEstimator(estimator=GaussianProcessClassifier())
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf = SubSampleEstimator(
            estimator=SklearnClassifier(GaussianProcessClassifier(), classes=["ny", "paris", "tokyo"], missing_label="nan"),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labled=True,
        )
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
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
        # Soll der einen Haben. Wenn ja erbe von sklearn ...
        self.assertFalse(hasattr(clf, "kernel_"))

        X = [[1], [0]]
        y_true = [1, 0]
        clf = SubSampleEstimator(
            SklearnClassifier(GaussianProcessClassifier()), classes=[0, 1]
        )
        ensemble = SubSampleEstimator(
            SklearnClassifier(BaggingClassifier(clf)), classes=[0, 1]
        )
        ensemble.fit(X, y_true)
        self.assertTrue(ensemble.is_fitted_, True)

    def test_partial_fit(self):
        clf = SubSampleEstimator(
            SklearnClassifier(estimator=GaussianNB(), missing_label="nan"),
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
        clf = SubSampleEstimator(
            estimator=SklearnClassifier(GaussianNB(), missing_label="nan"),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labled=True,
        )
        self.assertEqual(clf.missing_label, "nan")
        clf.partial_fit(self.X, self.y2, sample_weight=np.ones_like(self.y2))
        self.assertTrue(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        # is allways true since it simmulates partial_fit
        self.assertTrue(hasattr(clf, "partial_fit"))

    def test_predict_proba(self):
        clf = SubSampleEstimator(
            SklearnClassifier(estimator=GaussianProcessClassifier(), missing_label="nan"),
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
        clf = SubSampleEstimator(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan", classes=["ny", "paris", "tokyo"]
            ),
            classes=["ny", "paris", "tokyo"],
            missing_label="nan",
        )
        # Fehler da im try catch block eine fehlermelung ausgeworfen wird, wodurch
        # das programm abgebrochen wird was im normal test nicht passiert. Dadurch wird
        # n_classes nicht erstellt, was für predict proba verwendet wird
        # TODO Label encoder rausnehmen
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
        clf = SubSampleEstimator(
            estimator=SklearnClassifier(GaussianProcessClassifier(),missing_label="nan"),
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

