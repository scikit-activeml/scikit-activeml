import random
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
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils.validation import NotFittedError, check_is_fitted

from skactiveml.classifier import (
    SklearnClassifier,
    SlidingWindowClassifier,
    ParzenWindowClassifier,
    MixtureModelClassifier,
)
from skactiveml.tests.template_estimator import TemplateSkactivemlClassifier
from skactiveml.utils import MISSING_LABEL

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.classifier import SkorchClassifier
    from skorch.utils import to_numpy

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover


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
        test_cases = [
            (Perceptron(), None),
            ("Test", AttributeError),
            (GaussianNB(), None),
            (LinearRegression(), TypeError),
        ]
        self._test_param("init", "estimator", test_cases)

    def test_init_param_include_unlabeled_samples(self):
        test_cases = [
            (GaussianNB(), TypeError),
            (True, None),
            (False, None),
            ("String", TypeError),
        ]
        self._test_param("init", "include_unlabeled_samples", test_cases)

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

        # Test semi-supervised learning.
        X, y = make_blobs(
            centers=10, n_samples=200, random_state=0, shuffle=True
        )
        y_partial = np.full_like(y, -1)
        y_partial[:50] = y[:50]
        clf = SklearnClassifier(
            estimator=SelfTrainingClassifier(
                LogisticRegression(random_state=0),
                threshold=0,
                max_iter=10,
                verbose=True,
            ),
            include_unlabeled_samples=True,
            missing_label=-1,
            classes=np.unique(y),
            random_state=0,
        )
        clf.fit(X, y_partial)
        self.assertEqual((clf.labeled_iter_ > 0).sum(), 150)
        clf.set_params(include_unlabeled_samples=False)
        clf.fit(X, y_partial)
        self.assertEqual((clf.labeled_iter_ > 0).sum(), 0)

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
            # different classes than provided to the `classes` parameter of
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
            ([0, 1, 0], TypeError),
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
            ([0, 1, 1], None),
            (["tokyo", "nan", "paris"], TypeError),
        ]
        replace_init_params = {
            "classes": [0, 1],
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
            "classes": [0, 1],
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


if successful_skorch_torch_import:

    class TestSkorchClassifier(
        TemplateSkactivemlClassifier, unittest.TestCase
    ):
        def setUp(self):
            # Set global seeds.
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            self.X, self.y_true = make_blobs(
                n_samples=200, n_features=1, centers=2, random_state=0
            )
            self.X = self.X.astype(np.float32)
            self.y = np.copy(self.y_true).astype(np.float32)
            self.y[:100] = MISSING_LABEL
            self.y_ulbld = np.full_like(self.y, fill_value=MISSING_LABEL)
            self.classes = np.unique(self.y_true)

            estimator_class = SkorchClassifier
            self.neural_net_param_dict = {
                "train_split": None,
                "verbose": False,
                "optimizer": torch.optim.RAdam,
                "device": "cpu",
                "lr": 0.01,
                "max_epochs": 30,
                "batch_size": 2,
            }
            init_default_params = {
                "module": TestNeuralNet,
                "classes": None,
                "missing_label": MISSING_LABEL,
                "random_state": 1,
                "neural_net_param_dict": self.neural_net_param_dict,
            }
            fit_default_params = {
                "X": self.X,
                "y": self.y,
            }
            predict_default_params = {"X": self.X}
            super().setUp(
                estimator_class=estimator_class,
                init_default_params=init_default_params,
                fit_default_params=fit_default_params,
                predict_default_params=predict_default_params,
            )

        def test_init_param_module(self, test_cases=None):
            clf = SkorchClassifier(module="Test")
            self.assertEqual(clf.module, "Test")

            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                ("Test", TypeError),
                (None, TypeError),
                ([("nn.Module", TestNeuralNet)], TypeError),
            ]
            self._test_param("init", "module", test_cases)

        def test_init_param_criterion(self, test_cases=None):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                ("Test", TypeError),
                (None, TypeError),
                (nn.NLLLoss, None),
                (nn.CrossEntropyLoss, None),
                (nn.NLLLoss(), None),
                (nn.CrossEntropyLoss(), None),
            ]
            self._test_param("init", "criterion", test_cases)

        def test_init_param_include_unlabeled_samples(self, test_cases=None):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (GaussianNB(), TypeError),
                (True, IndexError),
                (False, None),
                ("String", TypeError),
            ]
            self._test_param("init", "include_unlabeled_samples", test_cases)
            neural_net_param_dict = self.neural_net_param_dict.copy()
            test_cases = [(True, None)]
            neural_net_param_dict["criterion__ignore_index"] = -1
            self._test_param(
                "init",
                "include_unlabeled_samples",
                test_cases,
                replace_init_params={
                    "neural_net_param_dict": neural_net_param_dict
                },
            )

        def test_initialize(self):
            # Check prediction w/o initialization and w/o given classes.
            clf = SkorchClassifier(**self.init_default_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            y_pred = clf.predict(self.X)
            self.assertTrue((np.isin(y_pred, self.classes)).all())

            # Check prediction with initialization and w/o given classes.
            clf = SkorchClassifier(**self.init_default_params)
            clf.initialize()
            y_pred = clf.predict(self.X)
            self.assertTrue((np.isin(y_pred, self.classes)).all())

        def test_fit(self):
            # Check standard fitting cases.
            clf = SkorchClassifier(**self.init_default_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            self.assertRaises(ValueError, clf.fit, self.X, self.y_ulbld)
            clf.fit(self.X, self.y)
            check_is_fitted(clf)

            # Check fitting without `warm_restart`.
            init_default_params1 = self.init_default_params.copy()
            init_default_params1["classes"] = [0, 1]
            init_default_params1["neural_net_param_dict"]["warm_start"] = False
            clf = SkorchClassifier(**init_default_params1)
            clf.fit(self.X, self.y_ulbld)
            init_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            clf.fit(self.X, self.y_ulbld)
            new_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Check fitting with `warm_restart`.
            init_default_params2 = self.init_default_params.copy()
            init_default_params2["classes"] = [0, 1]
            init_default_params2["neural_net_param_dict"]["warm_start"] = True
            init_default_params2["neural_net_param_dict"]["verbose"] = 1
            clf = SkorchClassifier(**init_default_params2)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf.fit(self.X, self.y_ulbld)
            check_is_fitted(clf)
            init_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            clf.fit(self.X, self.y_ulbld)
            new_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            np.testing.assert_array_equal(init_weights, new_weights)
            clf.fit(self.X, self.y)
            new_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Setup for initialized Pytorch module as input.
            init_default_params3 = self.init_default_params.copy()
            init_default_params3["classes"] = [0, 1]
            clf_module = TestNeuralNet()
            init_weights = to_numpy(
                deepcopy(clf_module.input_to_hidden.weight)
            )
            init_default_params3["module"] = clf_module
            clf = SkorchClassifier(**init_default_params3)

            # Fitting with only unlabeled data must preserve weights.
            clf.fit(self.X, self.y_ulbld)
            new_weights = to_numpy(deepcopy(clf_module.input_to_hidden.weight))
            np.testing.assert_array_equal(init_weights, new_weights)

            # Fitting with partially label data must change weights.
            clf.fit(self.X, self.y)
            new_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

        def test_partial_fit(self):
            clf = SkorchClassifier(**self.init_default_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            self.assertRaises(
                ValueError, clf.partial_fit, self.X, self.y_ulbld
            )
            clf.partial_fit(self.X, self.y)
            check_is_fitted(clf)

            init_default_params2 = self.init_default_params.copy()
            init_default_params2["classes"] = [0, 1]
            clf = SkorchClassifier(**init_default_params2)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf.partial_fit(self.X, self.y_ulbld)
            clf.partial_fit(self.X, self.y)
            check_is_fitted(clf)

            predict_proba_0 = clf.predict_proba(self.X)
            clf.partial_fit(self.X, self.y_ulbld)
            predict_proba_1 = clf.predict_proba(self.X)
            np.testing.assert_almost_equal(predict_proba_0, predict_proba_1)

        def test_predict(self):
            clf = SkorchClassifier(**self.init_default_params)
            clf.fit(**self.fit_default_params)
            y_pred = clf.predict(self.fit_default_params["X"])
            self.assertEqual(len(y_pred), len(self.X))

        def test_predict_proba(self):
            init_default_params = self.init_default_params.copy()
            init_default_params["forward_outputs"] = {
                "probas": (0, nn.Softmax(dim=-1)),
                "logits": (0, None),
                "emb": (1, None),
            }
            clf = SkorchClassifier(**init_default_params)
            clf.fit(self.X, self.y_true)
            P_class, L_class, X_embed = clf.predict_proba(
                self.X, extra_outputs=["logits", "emb"]
            )
            self.assertTrue((P_class.sum(axis=-1).round(3) == 1).all())
            self.assertTrue((P_class > 0).all())
            np.testing.assert_array_equal(L_class.shape, (len(self.X), 2))
            self.assertTrue((L_class < 0).any())
            self.assertTrue(X_embed.shape[1], 1)
            init_default_params = self.init_default_params.copy()
            init_default_params["classes"] = [0, 1]
            clf = SkorchClassifier(**init_default_params)
            P_class_0 = clf.predict_proba(self.X)
            clf.partial_fit(self.X, self.y_ulbld)
            P_class_1 = clf.predict_proba(self.X)
            np.testing.assert_almost_equal(P_class_0, P_class_1)
            clf.fit(self.X, self.y)
            P_class_2 = clf.predict_proba(self.X)
            self.assertEqual(len(P_class_2), len(self.X))
            self.assertEqual(P_class_2.shape[1], 2)

        def test_init_param_sample_dtype(self):
            test_cases = [
                (None, None),
                (np.float32, None),
                (np.int32, RuntimeError),
            ]
            self._test_param("init", "sample_dtype", test_cases)

        def test_init_param_neural_net_param_dict(self):
            default_dict = self.init_default_params["neural_net_param_dict"]
            test_cases = [
                (None, None),
                (default_dict, None),
                (default_dict, None),
                (np.int32, TypeError),
                ("a", TypeError),
                ({"abcdefg": 0}, ValueError),
                ({"predict_nonlinearity": nn.Identity()}, ValueError),
                ({"module": TestNeuralNet}, ValueError),
                ({"criterion": nn.CrossEntropyLoss}, ValueError),
            ]
            self._test_param("init", "neural_net_param_dict", test_cases)

        def test_init_param_forward_outputs(self):
            test_cases = [
                (None, None),
                ({"proba": (0, None)}, None),
                ({"proba": (0, None), "emb": (1, None)}, None),
                (
                    {
                        "proba": (0, nn.Softmax(dim=-1)),
                        "logits": (0, None),
                        "emb": (1, None),
                    },
                    None,
                ),
                ({"proba": (0,)}, TypeError),
                ({"proba": (-1, None)}, ValueError),
                ({"proba": ("str", None)}, TypeError),
                ({"proba": (2, None)}, ValueError),
            ]
            self._test_param("init", "forward_outputs", test_cases)

            test_cases = [
                (None, None),
                ({"proba": (0, torch.exp)}, None),
            ]
            self._test_param(
                "init",
                "forward_outputs",
                test_cases,
                replace_init_params={"criterion": nn.NLLLoss},
            )

        def test_init_param_criterion_output_keys(self):
            test_cases = [
                (None, None),
                ("proba", None),
                (["proba"], None),
                ("test", ValueError),
                (["test"], ValueError),
                (False, TypeError),
            ]
            self._test_param("init", "criterion_output_keys", test_cases)

            replace_init_params = {
                "forward_outputs": {
                    "proba": (0, nn.Softmax(dim=-1)),
                    "logits": (0, None),
                    "emb": (1, None),
                }
            }
            test_cases += [
                ("proba", None),
                (["proba"], None),
                ("emb", IndexError),
                (["emb"], IndexError),
                (["proba", "logits"], ValueError),
                (["logits", "emb"], TypeError),
            ]
            self._test_param(
                "init",
                "criterion_output_keys",
                test_cases,
                replace_init_params=replace_init_params,
            )
            nn_rep = self.init_default_params["neural_net_param_dict"].copy()
            nn_rep["module__return_embeddings"] = False
            test_cases = [
                (None, None),
                ("proba", None),
                (["proba"], None),
                ("test", ValueError),
                (["test"], ValueError),
                (False, TypeError),
            ]
            self._test_param(
                "init",
                "criterion_output_keys",
                test_cases,
                replace_init_params={"neural_net_param_dict": nn_rep},
            )

        def test_predict_param_extra_outputs(self):
            self._test_extra_outputs("predict_proba")

        def test_predict_proba_param_extra_outputs(self):
            self._test_extra_outputs("predict_proba")

        def _test_extra_outputs(self, predict_method):
            test_cases = [
                (None, None),
                ([], None),
                ("proba", ValueError),
                (["proba"], ValueError),
                ("emb", ValueError),
                (["emb"], ValueError),
                ("logits", ValueError),
                (["logits", "emb"], ValueError),
                (["emb", "logits"], ValueError),
                (False, TypeError),
            ]
            self._test_param(
                predict_method,
                "extra_outputs",
                test_cases,
                extras_params={"X": self.X},
            )
            test_cases = [
                (None, None),
                ([], None),
                ("proba", ValueError),
                (["proba"], ValueError),
                ("emb", None),
                (["emb"], None),
                ("logits", None),
                (["logits", "emb"], None),
                (["emb", "logits"], None),
                (False, TypeError),
            ]
            replace_init_params = {
                "forward_outputs": {
                    "proba": (0, nn.Softmax(dim=-1)),
                    "logits": (0, None),
                    "emb": (1, None),
                }
            }
            self._test_param(
                predict_method,
                "extra_outputs",
                test_cases,
                extras_params={"X": self.X},
                replace_init_params=replace_init_params,
            )

    class TestNeuralNet(nn.Module):
        def __init__(self, return_embeddings=True):
            super().__init__()
            self.return_embeddings = return_embeddings
            self.input_to_hidden = nn.Linear(
                in_features=1, out_features=1, bias=True, dtype=torch.float32
            )
            self.hidden_to_output = nn.Linear(
                in_features=1, out_features=2, bias=True, dtype=torch.float32
            )

        def forward(self, X):
            hidden = self.input_to_hidden(X)
            hidden = torch.sigmoid(hidden)
            output_values = self.hidden_to_output(hidden)
            if self.return_embeddings:
                return output_values, hidden
            else:
                return output_values
