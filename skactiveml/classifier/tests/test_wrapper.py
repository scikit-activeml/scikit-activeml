import unittest
import warnings

from copy import deepcopy
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Perceptron,
    SGDClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import NotFittedError, check_is_fitted

from skactiveml.classifier import (
    SklearnClassifier,
    SlidingWindowClassifier,
    ParzenWindowClassifier,
    MixtureModelClassifier,
)
from skactiveml.tests.template_estimator import TemplateSkactivemlClassifier


class TestSklearnClassifier(TemplateSkactivemlClassifier, unittest.TestCase):
    def setUp(self):
        estimator_class = SklearnClassifier
        init_default_params = {
            "estimator": GaussianNB(),
            "missing_label": "nan",
        }
        fit_default_params = {
            "X": np.zeros((4, 1)),
            "y": ["tokyo", "paris", "nan", "tokyo"],
        }
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )

        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    def test_init_param_estimator(self):
        test_cases = []
        test_cases += [
            (Perceptron(), None),
            ("Test", AttributeError),
            (GaussianNB(), None),
            (LinearRegression(), TypeError),
        ]
        self._test_param("init", "estimator", test_cases)

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
        self.assertRaises(
            ValueError,
            clf.fit,
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
        )
        clf = SklearnClassifier(estimator=GaussianProcessClassifier())
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(NotFittedError, check_is_fitted, estimator=clf)
        clf.fit(
            self.fit_default_params["X"],
            self.fit_default_params["y"],
        )
        self.assertTrue(clf.is_fitted_)
        clf.fit(self.fit_default_params["X"], self.fit_default_params["y"])
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "kernel_"))
        np.testing.assert_array_equal(
            clf.classes_, ["new york", "paris", "tokyo"]
        )
        self.assertEqual(clf.missing_label, "nan")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(self.fit_default_params["X"], self.y2)
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
        clf.partial_fit(
            self.fit_default_params["X"], self.fit_default_params["y"]
        )
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "class_count_"))
        np.testing.assert_array_equal(
            clf.classes_, ["new york", "paris", "tokyo"]
        )
        self.assertEqual(clf.missing_label, "nan")
        clf.partial_fit(
            self.fit_default_params["X"],
            self.y2,
            sample_weight=np.ones_like(self.y2),
        )
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
        self.assertRaises(
            NotFittedError, clf.predict_proba, X=self.fit_default_params["X"]
        )
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        P = clf.predict_proba(X=self.fit_default_params["X"])
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        P_exp = est.predict_proba(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(P_exp, P)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.fit_default_params["X"], y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P = clf.predict_proba(X=self.fit_default_params["X"])
            self.assertEqual(len(w), 1)
        P_exp = np.ones((len(self.fit_default_params["X"]), 1))
        np.testing.assert_array_equal(P_exp, P)
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            classes=["ny", "paris", "tokyo"],
            missing_label="nan",
        )
        clf.fit(X=self.fit_default_params["X"], y=self.y_nan)
        P = clf.predict_proba(X=self.fit_default_params["X"])
        P_exp = np.ones((len(self.fit_default_params["X"]), 3)) / 3
        np.testing.assert_array_equal(P_exp, P)
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        P = clf.predict_proba(X=self.fit_default_params["X"])
        P_exp = np.zeros((len(self.fit_default_params["X"]), 3))
        P_exp[:, 1:] = est.predict_proba(X=self.fit_default_params["X"])
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
        self.assertRaises(
            NotFittedError, clf.predict, X=self.fit_default_params["X"]
        )
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        y = clf.predict(X=self.fit_default_params["X"])
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        y_exp = est.predict(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(y, y_exp)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.fit_default_params["X"], y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = clf.predict(X=self.fit_default_params["X"])
            self.assertEqual(len(w), 1)
        y_exp = ["tokyo"] * len(self.fit_default_params["X"])
        np.testing.assert_array_equal(y_exp, y)

    def test_pipeline(self):
        X, y_true = make_blobs(100, centers=2, random_state=0)
        pipline = Pipeline(
            (
                ("scaler", StandardScaler()),
                ("gpc", GaussianProcessClassifier(random_state=0)),
            )
        )
        clf = SklearnClassifier(
            pipline, classes=[0, 1], missing_label=-1, random_state=0
        )
        clf = clf.fit(X, y_true)
        self.assertTrue(clf.is_fitted_)
        check_is_fitted(clf)
        self.assertRaises(NotFittedError, check_is_fitted, pipline)
        self.assertGreaterEqual(clf.score(X, y_true), 0.9)
        y_missing = np.full_like(y_true, -1)
        clf.fit(X, y_missing)
        self.assertFalse(clf.is_fitted_)
        check_is_fitted(clf)
        p = clf.predict_proba(X)
        np.testing.assert_array_equal(np.full_like(p, 0.5), p)

    def test_pretrained_estimator(self):
        random_state = np.random.RandomState(0)
        X_full, y_full = make_blobs(150, centers=2, random_state=0)
        X_train = X_full[:100]
        y_train_true = y_full[:100]
        X_test = X_full[100:]
        # y_test_true = X_full[100:]
        class_names = ["No", "Yes"]

        cases = [([0, 1], np.nan), (class_names, "None")]

        for class_mapping, missing_label in cases:
            y_train = np.array([class_mapping[y] for y in y_train_true])

            # pretrain classifier and test consistency of results after
            # wrapping
            pretrained_estimator = SGDClassifier(
                loss="modified_huber",
                random_state=0,
            )
            pretrained_estimator.fit(X_train, y_train)

            pred_proba_orig_0 = pretrained_estimator.predict_proba(X_test)
            pred_orig_0 = pretrained_estimator.predict(X_test)

            clf = SklearnClassifier(
                estimator=pretrained_estimator,
                missing_label=missing_label,
                classes=class_mapping,
                random_state=0,
            )

            pred_proba_wrapped_0 = clf.predict_proba(X_test)
            pred_wrapped_0 = clf.predict(X_test)

            np.testing.assert_array_equal(
                pred_proba_orig_0, pred_proba_wrapped_0
            )
            np.testing.assert_array_equal(pred_orig_0, pred_wrapped_0)

            # update classifier and check results for consistency afterwards
            y_train_random = random_state.permutation(y_train)

            pretrained_estimator.partial_fit(X_train, y_train_random)
            clf.partial_fit(X_train, y_train_random)

            pred_proba_orig_1 = pretrained_estimator.predict_proba(X_test)
            pred_orig_1 = pretrained_estimator.predict(X_test)
            pred_proba_wrapped_1 = clf.predict_proba(X_test)
            pred_wrapped_1 = clf.predict(X_test)

            np.testing.assert_array_equal(
                pred_proba_orig_1, pred_proba_wrapped_1
            )
            np.testing.assert_array_equal(pred_orig_1, pred_wrapped_1)

            # check that it fails when classes of estimator was trained on
            # different classes than profided to the `classes` parameter of
            # SklearnClassifier
            if not isinstance(missing_label, float):
                self.assertRaises(TypeError, clf.fit, X_train, y_train_true)
                self.assertRaises(TypeError, clf.fit, X_train, y_train_true)

        pretrained_estimator = SGDClassifier(
            loss="modified_huber",
            random_state=0,
        )
        pretrained_estimator.fit(X_train, y_train_true)
        clf = SklearnClassifier(
            estimator=pretrained_estimator,
            missing_label=np.nan,
            random_state=0,
            classes=[2, 3],
        )

        self.assertRaises(ValueError, clf.fit, X_train, y_train_true)

        self.assertRaises(ValueError, clf.partial_fit, X_train, y_train_true)


class TestSlidingWindowClassifier(
    TemplateSkactivemlClassifier, unittest.TestCase
):
    def setUp(self):
        estimator_class = SlidingWindowClassifier
        init_default_params = {
            "estimator": SklearnClassifier(
                SGDClassifier(loss="log_loss"),
                classes=["tokyo", "paris"],
                missing_label="nan",
            ),
            "missing_label": "nan",
        }
        fit_default_params = {
            "X": np.zeros((4, 1)),
            "y": ["tokyo", "paris", "nan", "tokyo"],
        }
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )

        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    def test_init_param_estimator(self):
        test_cases = []
        test_cases += [
            (ParzenWindowClassifier(missing_label="nan"), None),
            ("Test", AttributeError),
            (GaussianNB(), TypeError),
        ]
        self._test_param("init", "estimator", test_cases)
        clf = SlidingWindowClassifier(estimator=Perceptron())
        self.assertRaises(TypeError, clf.partial_fit, [[0], [1]], [[0], [1]])

    def test_init_param_missing_label(self, test_cases=None):
        replace_init_params = {
            "estimator": SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan"
            )
        }
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, TypeError), ("nan", None), (1, TypeError)]
        replace_init_params["classes"] = ["tokyo", "paris"]
        replace_fit_params = {
            "y": ["tokyo", "nan", "paris"],
            "X": np.zeros((3, 1)),
        }
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

        test_cases = [("state", TypeError), (-1, None), (-2, ValueError)]
        replace_init_params["classes"] = [0, 1]
        replace_init_params["estimator"] = SklearnClassifier(
            LogisticRegression(), missing_label=-1
        )
        replace_fit_params = {"y": [0, -1, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

        test_cases = [("state", TypeError), (None, None)]
        replace_init_params["classes"] = [0, 1]
        replace_init_params["estimator"] = SklearnClassifier(
            LogisticRegression(), missing_label=None
        )
        replace_fit_params = {"y": [0, None, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

        test_cases = [("state", TypeError), (0.0, None)]
        replace_init_params["classes"] = [0.5, 1.4]
        replace_init_params["estimator"] = SklearnClassifier(
            LogisticRegression(), missing_label=0.0
        )
        replace_fit_params = {"y": [0.5, 0, 1.4], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, TypeError),
            ([1, 2], TypeError),
            (["tokyo", "paris"], None),
            (["tokyo", "berlin"], ValueError),
        ]
        replace_init_params = {
            "estimator": SklearnClassifier(
                LogisticRegression(),
                missing_label="nan",
                classes=["tokyo", "paris"],
            )
        }
        replace_init_params = {"missing_label": "nan"}
        replace_fit_params = {
            "y": ["tokyo", "nan", "paris"],
            "X": np.zeros((3, 1)),
        }
        self._test_param(
            "init",
            "classes",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [([1, 2], None), (["tokyo", "paris"], TypeError)]
        replace_init_params = {"missing_label": -1}
        replace_init_params["estimator"] = SklearnClassifier(
            LogisticRegression(), missing_label=-1
        )
        replace_fit_params = {"y": [2, -1, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "classes",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_init_param_cost_matrix(self):
        super().test_init_param_cost_matrix()
        estimator = ParzenWindowClassifier(
            classes=[0, 1], cost_matrix=np.eye(2)
        )
        clf = SlidingWindowClassifier(
            estimator=estimator, classes=[0, 1], cost_matrix=2 * np.eye(2)
        )
        self.assertRaises(ValueError, clf.fit, [[0], [1]], [0, 1])

    def test_fit_param_X(self, test_cases=None, replace_init_params=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, ValueError),
            ([1], ValueError),
            (np.zeros((len(self.fit_default_params["y"]), 1)), None),
        ]
        self._test_param("fit", "X", test_cases)

        replace_init_params = {
            "estimator": MixtureModelClassifier(
                missing_label=-1, classes=[0, 1]
            )
        }
        test_cases = [([], None)]
        replace_fit_params = {"y": []}
        if replace_init_params is None:
            replace_init_params = {}
        replace_init_params["classes"] = [0, 1]
        replace_init_params["missing_label"] = -1
        self._test_param(
            "fit",
            "X",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [([], ValueError)]
        replace_init_params["classes"] = None
        replace_init_params["estimator"] = MixtureModelClassifier(
            missing_label=-1, classes=None
        )
        self._test_param(
            "fit",
            "X",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ([0, 1, 2], TypeError),
            (["tokyo", "nan", "paris"], None),
        ]
        replace_init_params = {
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
            "estimator": SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan"
            ),
        }
        replace_fit_params = {"X": np.zeros((3, 1))}
        self._test_param(
            "fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [
            ([0, 1, 2], None),
            (["tokyo", "nan", "paris"], TypeError),
        ]
        replace_init_params = {
            "classes": [0, 1, 2],
            "missing_label": -1,
            "estimator": SklearnClassifier(
                GaussianProcessClassifier(), missing_label=-1
            ),
        }
        replace_fit_params = {"X": np.zeros((3, 1))}
        self._test_param(
            "fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_partial_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ([0, 1, 2, -1], TypeError),
            (["tokyo"], ValueError),
            (["nan", "tokyo", "nan", "paris"], None),
        ]
        replace_init_params = {
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
            "estimator": SklearnClassifier(GaussianNB(), missing_label="nan"),
        }
        replace_fit_params = {"X": np.zeros((3, 1))}
        extras_params = deepcopy(self.fit_default_params)
        self._test_param(
            "partial_fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
            extras_params=extras_params,
            exclude_fit=True,
        )
        test_cases = [
            ([0, 1, 2, -1], None),
            (["nan", "nan", "nan", "nan"], TypeError),
        ]
        replace_init_params = {
            "classes": [0, 1, 2],
            "missing_label": -1,
            "estimator": SklearnClassifier(GaussianNB(), missing_label=-1),
        }
        replace_fit_params = {"X": np.zeros((3, 1))}
        self._test_param(
            "partial_fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
            extras_params=extras_params,
            exclude_fit=True,
        )

    def test_init_param_window_size(self):
        test_cases = []
        test_cases += [(100, None), (-1, ValueError), ("Test", TypeError)]
        self._test_param("init", "window_size", test_cases)

    def test_init_param_only_labeled(self):
        test_cases = []
        test_cases += [
            (True, None),
            (False, None),
            ("Test", TypeError),
            (0, TypeError),
        ]
        self._test_param("init", "only_labeled", test_cases)

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
        self.assertRaises(
            ValueError,
            clf.fit,
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
        )

        clf = SlidingWindowClassifier(estimator=GaussianNB())
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
        clf.fit(self.fit_default_params["X"], self.fit_default_params["y"])
        self.assertTrue(clf.is_fitted_)
        self.assertTrue(hasattr(clf, "kernel_"))
        np.testing.assert_array_equal(
            clf.estimator_.classes_, ["new york", "paris", "tokyo"]
        )
        self.assertEqual(clf.missing_label, "nan")
        # test if warnings are correctly handeled
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(self.fit_default_params["X"], self.y2)
            self.assertEqual(len(w), 1)
        self.assertFalse(clf.is_fitted_)
        self.assertFalse(clf.estimator_.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        # fit clf with no prior classes and no labels
        clf = SlidingWindowClassifier(
            SklearnClassifier(GaussianNB(), missing_label="nan"),
            missing_label="nan",
        )
        self.assertRaises(
            ValueError, clf.fit, X=self.fit_default_params["X"], y=self.y_nan
        )
        # fit clf with correct data and sample_weight
        clf = SlidingWindowClassifier(
            SklearnClassifier(GaussianNB(), missing_label="nan"),
            missing_label="nan",
        )
        clf.fit(
            self.fit_default_params["X"],
            self.fit_default_params["y"],
            sample_weight=np.ones(len(self.fit_default_params["y"])),
        )

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
        clf.partial_fit(
            self.fit_default_params["X"], self.fit_default_params["y"]
        )
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
        self.assertTrue(hasattr(clf, "partial_fit"))
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                Perceptron(),
                missing_label="nan",
                classes=["tokyo", "paris", "new york"],
            ),
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            only_labeled=True,
            window_size=5,
        )
        clf.partial_fit(
            self.fit_default_params["X"],
            self.fit_default_params["y"],
            sample_weight=np.ones_like(self.fit_default_params["y"]),
        )
        self.assertTrue(clf.is_fitted_)

        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(),
                classes=["tokyo", "paris", "new york"],
                missing_label="nan",
            )
        )
        self.assertTrue(hasattr(clf, "partial_fit"))

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
            self.fit_default_params["X"],
            self.y_nan,
            sample_weight=np.ones_like(self.y_nan),
        )
        clf.partial_fit(
            self.fit_default_params["X"],
            self.y2,
            sample_weight=np.ones_like(self.y2),
        )
        self.assertTrue(clf.is_fitted_)
        self.assertFalse(hasattr(clf, "kernel_"))
        clf.partial_fit(
            self.fit_default_params["X"],
            self.y2,
            sample_weight=np.ones_like(self.y2),
        )
        self.assertEqual(len(clf.X_train_), 5)
        clf.partial_fit(
            self.fit_default_params["X"],
            self.y_nan,
            sample_weight=np.ones_like(self.y2),
        )
        # test clf with classes and empty data
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                Perceptron(),
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
            self.fit_default_params["X"],
            self.y_nan,
            sample_weight=np.ones_like(self.y2),
        )
        y2 = np.array(["tokyo", "nan", "nan", "paris"])
        clf.partial_fit(
            self.fit_default_params["X"],
            y2,
            sample_weight=np.ones_like(y2, dtype=float),
        )
        self.assertTrue(clf.is_fitted_)

    def test_predict_proba(self):
        clf = SlidingWindowClassifier(
            SklearnClassifier(
                estimator=GaussianProcessClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        self.assertRaises(
            NotFittedError, clf.predict_proba, X=self.fit_default_params["X"]
        )
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        P = clf.predict_proba(X=self.fit_default_params["X"])
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        P_exp = est.predict_proba(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(P_exp, P)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.fit_default_params["X"], y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P = clf.predict_proba(X=self.fit_default_params["X"])
            self.assertEqual(len(w), 1)
        P_exp = np.ones((len(self.fit_default_params["X"]), 1))
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
        clf.fit(X=self.fit_default_params["X"], y=self.y_nan)
        P = clf.predict_proba(X=self.fit_default_params["X"])
        P_exp = np.ones((len(self.fit_default_params["X"]), 3)) / 3
        np.testing.assert_array_equal(P_exp, P)
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        P = clf.predict_proba(X=self.fit_default_params["X"])
        P_exp = np.zeros((len(self.fit_default_params["X"]), 3))
        P_exp[:, 1:] = est.predict_proba(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(P_exp, P)

    def test_predict(self):
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianProcessClassifier(), missing_label="nan"
            ),
            missing_label="nan",
        )
        self.assertRaises(
            NotFittedError, clf.predict, X=self.fit_default_params["X"]
        )
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        y = clf.predict(X=self.fit_default_params["X"])
        est = GaussianProcessClassifier().fit(
            X=np.zeros((3, 1)), y=["tokyo", "paris", "tokyo"]
        )
        y_exp = est.predict(X=self.fit_default_params["X"])
        # Predicts wrong classes (numbers instead of strings)
        np.testing.assert_array_equal(y, y_exp)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.fit_default_params["X"], y=self.y2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = clf.predict(X=self.fit_default_params["X"])
            self.assertEqual(len(w), 1)
        y_exp = ["tokyo"] * len(self.fit_default_params["X"])
        np.testing.assert_array_equal(y_exp, y)

    def test_predict_freq(self):
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(missing_label="nan"),
            missing_label="nan",
        )
        self.assertRaises(
            NotFittedError, clf.predict_freq, X=self.fit_default_params["X"]
        )
        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        freq = clf.predict_freq(X=self.fit_default_params["X"])

        self.assertEqual(len(np.unique(freq)), 2)
        est = ParzenWindowClassifier(missing_label="nan").fit(
            X=self.fit_default_params["X"], y=self.fit_default_params["y"]
        )
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(missing_label="nan"),
            missing_label="nan",
        )

        clf.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        freq = clf.predict_freq(X=self.fit_default_params["X"])
        est.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        freq_est = est.predict_freq(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(freq, freq_est)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
