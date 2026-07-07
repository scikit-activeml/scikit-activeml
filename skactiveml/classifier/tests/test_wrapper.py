import random
import unittest
import warnings
import inspect

from copy import deepcopy
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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
from sklearn.multioutput import MultiOutputClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.model_selection import train_test_split

from skactiveml.classifier import (
    SklearnClassifier,
    SlidingWindowClassifier,
    ParzenWindowClassifier,
    MixtureModelClassifier,
)
from skactiveml.classifier._wrapper import _prior_matrix_from_counts
from skactiveml.tests.template_estimator import TemplateSkactivemlClassifier
from skactiveml.utils import MISSING_LABEL

import importlib.util

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.classifier import SkorchClassifier
    from skorch.utils import to_numpy

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover

successful_river_import = False
try:
    from skactiveml.classifier import RiverClassifier
    import pandas as pd
    import river.tree
    import river.neighbors
    import river.naive_bayes
    import river.multiclass
    import river.forest
    import river.linear_model

    successful_river_import = True
except ImportError:
    pass  # pragma: no cover

spec = importlib.util.find_spec("capymoa")
successful_capymoa_import = spec is not None


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
        self.X_ml = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
        self.y_ml = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=int)
        init_default_params_multilabel = {
            "estimator": MultiOutputClassifier(
                SGDClassifier(loss="log_loss", random_state=0)
            ),
            "classes": [[0, 1], [0, 1]],
            "missing_label": -1,
            "proba_format": "array",
        }
        fit_default_params_multilabel = {
            "X": self.X_ml,
            "y": self.y_ml,
        }
        predict_def_params_multilabel = {"X": self.X_ml}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
            init_default_params_multilabel=init_default_params_multilabel,
            fit_default_params_multilabel=fit_default_params_multilabel,
            predict_default_params_multilabel=predict_def_params_multilabel,
        )

        self.y2 = ["tokyo", "nan", "nan", "tokyo"]
        self.y_nan = ["nan", "nan", "nan", "nan"]

    class _NoTargetEstimator:
        def fit(self, X, z):
            return self

    class _PredictProbaEstimator:
        def __init__(self, proba, classes_=None, estimators_=None):
            self._proba = proba
            if classes_ is not None:
                self.classes_ = classes_
            if estimators_ is not None:
                self.estimators_ = estimators_

        def predict_proba(self, X, **kwargs):
            return self._proba

        def predict(self, X, **kwargs):
            return np.zeros((len(X),), dtype=int)

    @staticmethod
    def _prefit_multilabel_clf(proba_format="array", classes=None):
        if classes is None:
            classes = [[0, 1], [0, 1]]
        clf = SklearnClassifier(
            estimator=GaussianNB(),
            classes=classes,
            missing_label=-1,
            proba_format=proba_format,
            random_state=0,
        )
        clf.check_X_dict_ = {"ensure_min_samples": 0, "ensure_min_features": 0}
        clf.n_features_in_ = 1
        dummy_classes = np.array([[classes[0][0], classes[1][0]]], dtype=int)
        clf._initialize_label_state(dummy_classes)
        clf.is_fitted_ = True
        return clf

    def test_prior_matrix_from_counts(self):
        np.testing.assert_allclose(
            _prior_matrix_from_counts([0, 0], n_samples=3),
            np.full((3, 2), 0.5),
        )
        np.testing.assert_allclose(
            _prior_matrix_from_counts([1, 3], n_samples=2),
            np.array([[0.25, 0.75], [0.25, 0.75]]),
        )

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

    def test_init_param_proba_format(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("auto", None), ("list", None), ("array", None)]
        self._test_param("init", "proba_format", test_cases)

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

    def test_multilabel_predict_proba(self):
        X = self.X_ml
        y = self.y_ml
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            proba_format="array",
        )

        clf.fit(X=X, Y=y)
        self.assertTrue(clf.is_fitted_)
        P = clf.predict_proba(X)
        self.assertEqual(P.shape, (len(X), 2))
        y_pred = clf.predict(X)
        self.assertEqual(y_pred.shape, y.shape)

    def test_multilabel_signature_uses_Y(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            proba_format="array",
        )
        fit_signature = inspect.signature(clf.fit)
        self.assertIn("Y", fit_signature.parameters)
        self.assertNotIn("y", fit_signature.parameters)
        self.assertRaises(TypeError, clf.fit, X=self.X_ml, y=self.y_ml)
        clf.fit(X=self.X_ml, Y=self.y_ml)
        self.assertTrue(clf.is_fitted_)

    def test_prefit_multilabel_infers_classes(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(estimator=estimator, classes=None)

        P = clf.predict_proba(self.X_ml)
        y_pred = clf.predict(self.X_ml)

        self.assertTrue(clf.multioutput_)
        for classes, expected_classes in zip(clf.classes_, estimator.classes_):
            np.testing.assert_array_equal(classes, expected_classes)
        self.assertEqual(P.shape, self.y_ml.shape)
        self.assertEqual(y_pred.shape, self.y_ml.shape)

        clf = SklearnClassifier(
            estimator=estimator, classes=None, proba_format="list"
        )
        P_list = clf.predict_proba(self.X_ml)

        self.assertTrue(clf.multioutput_)
        self.assertEqual(len(P_list), self.y_ml.shape[1])
        for P_j in P_list:
            self.assertEqual(P_j.shape, (len(self.X_ml), 2))

    def test_prefit_single_output_infers_classes(self):
        estimator = GaussianNB().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(estimator=estimator, classes=None)

        P = clf.predict_proba(self.X_ml)

        self.assertFalse(clf.multioutput_)
        np.testing.assert_array_equal(clf.classes_, estimator.classes_)
        self.assertEqual(P.shape, (len(self.X_ml), len(estimator.classes_)))

    def test_prefit_single_output_nan_falls_back_to_uniform_prior(self):
        estimator = GaussianNB().fit(self.X_ml, self.y_ml[:, 0])

        def predict_proba_nan(X, **kwargs):
            return np.full((len(X), len(estimator.classes_)), np.nan)

        estimator.predict_proba = predict_proba_nan
        clf = SklearnClassifier(
            estimator=estimator, classes=None, missing_label=-1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = clf.predict_proba(self.X_ml)

        self.assertEqual(P.shape, (len(self.X_ml), len(estimator.classes_)))
        self.assertFalse(np.any(np.isnan(P)))
        np.testing.assert_allclose(P, np.full_like(P, 0.5))

    def test_prefit_multilabel_nan_array_falls_back_to_uniform_prior(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )

        def predict_proba_nan(X, **kwargs):
            return [
                np.full((len(X), len(classes_j)), np.nan)
                for classes_j in estimator.classes_
            ]

        estimator.predict_proba = predict_proba_nan
        clf = SklearnClassifier(
            estimator=estimator,
            classes=None,
            missing_label=-1,
            proba_format="array",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = clf.predict_proba(self.X_ml)

        self.assertEqual(P.shape, self.y_ml.shape)
        self.assertFalse(np.any(np.isnan(P)))
        np.testing.assert_allclose(P, np.full_like(P, 0.5))

    def test_prefit_multilabel_nan_list_falls_back_to_uniform_prior(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )

        def predict_proba_nan(X, **kwargs):
            return [
                np.full((len(X), len(classes_j)), np.nan)
                for classes_j in estimator.classes_
            ]

        estimator.predict_proba = predict_proba_nan
        clf = SklearnClassifier(
            estimator=estimator,
            classes=None,
            missing_label=-1,
            proba_format="list",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P_list = clf.predict_proba(self.X_ml)

        self.assertEqual(len(P_list), self.y_ml.shape[1])
        for P_j in P_list:
            self.assertEqual(P_j.shape, (len(self.X_ml), 2))
            self.assertFalse(np.any(np.isnan(P_j)))
            np.testing.assert_allclose(P_j, np.full_like(P_j, 0.5))

    def test_multilabel_predict_proba_ensemble_estimators_collision(self):
        # A native multilabel ensemble stores base learners in `estimators_`,
        # so `len(estimators_)` must not be mistaken for the number of outputs
        # when it happens to coincide with it (regression test).
        n_outputs = self.y_ml.shape[1]
        for n_estimators in (n_outputs, n_outputs + 3):
            estimator = RandomForestClassifier(
                n_estimators=n_estimators, random_state=0
            ).fit(self.X_ml, self.y_ml)
            expected = estimator.predict_proba(self.X_ml)

            clf = SklearnClassifier(
                estimator=estimator,
                classes=[[0, 1], [0, 1]],
                missing_label=-1,
                proba_format="list",
            )
            P_list = clf.predict_proba(self.X_ml)
            self.assertEqual(len(P_list), n_outputs)
            for j, P_j in enumerate(P_list):
                self.assertEqual(P_j.shape, (len(self.X_ml), 2))
                np.testing.assert_allclose(P_j, expected[j])

            clf = SklearnClassifier(
                estimator=estimator,
                classes=[[0, 1], [0, 1]],
                missing_label=-1,
                proba_format="array",
            )
            P_array = clf.predict_proba(self.X_ml)
            self.assertEqual(P_array.shape, (len(self.X_ml), n_outputs))
            for j in range(n_outputs):
                np.testing.assert_allclose(P_array[:, j], expected[j][:, 1])

    def test_multilabel_predict_proba_array_nan_falls_back_to_prior(self):
        # A fitted estimator that yields NaN probabilities in `array` format
        # must degrade to the label-count prior instead of raising.
        n_samples = len(self.X_ml)
        clf = self._prefit_multilabel_clf(proba_format="array")
        clf._label_counts = [np.array([1, 3]), np.array([3, 1])]
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[
                np.full((n_samples, 2), np.nan),
                np.full((n_samples, 2), 0.5),
            ],
            classes_=[np.array([0, 1]), np.array([0, 1])],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = clf.predict_proba(self.X_ml)
        self.assertEqual(P.shape, (n_samples, 2))
        self.assertFalse(np.any(np.isnan(P)))
        np.testing.assert_allclose(P[:, 0], np.full(n_samples, 0.75))
        np.testing.assert_allclose(P[:, 1], np.full(n_samples, 0.25))

    def test_multilabel_predict_proba_unmappable_columns_raise(self):
        # An estimator exposing neither `classes_` nor per-output
        # `estimators_` that returns fewer columns than declared classes for
        # an output cannot be mapped, so a clear error is raised.
        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[
                np.ones((len(self.X_ml), 1)),
                np.full((len(self.X_ml), 2), 0.5),
            ]
        )
        self.assertRaisesRegex(
            ValueError,
            r"P\[0\] has 1 columns but output 0 declares 2 classes",
            clf.predict_proba,
            self.X_ml,
        )

    def test_helper_methods_and_prefit_sampling(self):
        clf = SklearnClassifier(estimator=GaussianNB(), missing_label=-1)
        self.assertRaises(
            TypeError,
            clf._extract_target_arg,
            self.y_ml,
            {"Y": self.y_ml},
        )

        estimator = self._NoTargetEstimator()
        self.assertRaises(TypeError, clf._target_parameter_name, estimator.fit)

        clf = self._prefit_multilabel_clf(proba_format="array")
        clf.is_fitted_ = False
        clf._label_counts = [np.array([1, 3]), np.array([3, 1])]
        y_pred = clf.predict(np.zeros((4, 1)))
        self.assertEqual(y_pred.shape, (4, 2))

        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.is_fitted_ = False
        clf._label_counts = [np.array([1, 3]), np.array([3, 1])]
        y_pred = clf.predict(np.zeros((4, 1)))
        self.assertEqual(y_pred.shape, (4, 2))

    def test_multilabel_predict_proba_edge_cases(self):
        n_samples = len(self.X_ml)

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones((n_samples, 2))], classes_=[np.array([0, 1])]
        )
        self.assertRaises(ValueError, clf.predict_proba, self.X_ml)

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones(n_samples), np.ones((n_samples, 2))],
            classes_=[np.array([0, 1]), np.array([0, 1])],
        )
        self.assertRaises(ValueError, clf.predict_proba, self.X_ml)

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones((n_samples - 1, 2)), np.ones((n_samples, 2))],
            classes_=[np.array([0, 1]), np.array([0, 1])],
        )
        self.assertRaisesRegex(
            ValueError,
            "Expected P\\[0\\] to contain 4 samples, got 3",
            clf.predict_proba,
            self.X_ml,
        )

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones((n_samples, 1)), np.ones((n_samples, 2))],
            classes_=[np.array([0, 1]), np.array([0, 1])],
        )
        self.assertRaisesRegex(
            ValueError,
            "P\\[0\\] has 1 columns but the fitted estimator reports "
            "2 classes",
            clf.predict_proba,
            self.X_ml,
        )

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones((n_samples, 1)), np.ones((n_samples, 1))],
            classes_=[np.array([1]), np.array([0])],
        )
        P = clf.predict_proba(self.X_ml)
        np.testing.assert_array_equal(P[:, 0], np.ones(n_samples))
        np.testing.assert_array_equal(P[:, 1], np.zeros(n_samples))

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones((n_samples, 2)), np.ones((n_samples, 2))],
            classes_=[np.array([2, 3]), np.array([0, 1])],
        )
        self.assertRaises(ValueError, clf.predict_proba, self.X_ml)

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.ones((n_samples, 3))
        )
        self.assertRaises(ValueError, clf.predict_proba, self.X_ml)

        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((n_samples, 2), 0.5)
        )
        P = clf.predict_proba(self.X_ml)
        self.assertEqual(P.shape, (n_samples, 2))

        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.full((n_samples, 2), 0.5), np.full((n_samples, 2), 0.5)]
        )
        P_list = clf.predict_proba(self.X_ml)
        self.assertEqual(len(P_list), 2)
        self.assertEqual(P_list[0].shape, (n_samples, 2))

        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.estimator_ = self._PredictProbaEstimator(
            proba=[np.ones((n_samples, 1)), np.ones((n_samples, 1))],
            classes_=[np.array([1]), np.array([0])],
        )
        P_list = clf.predict_proba(self.X_ml)
        self.assertEqual(len(P_list), 2)
        self.assertEqual(P_list[0].shape, (n_samples, 2))
        np.testing.assert_array_equal(
            P_list[0], np.tile([0.0, 1.0], (n_samples, 1))
        )
        np.testing.assert_array_equal(
            P_list[1], np.tile([1.0, 0.0], (n_samples, 1))
        )

        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((n_samples, 2), 0.5)
        )
        P_list = clf.predict_proba(self.X_ml)
        self.assertEqual(len(P_list), 2)
        self.assertEqual(P_list[0].shape, (n_samples, 2))

        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.is_fitted_ = False
        clf._label_counts = [np.array([1, 3]), np.array([3, 1])]
        P_list = clf.predict_proba(self.X_ml)
        self.assertEqual(len(P_list), 2)
        self.assertEqual(P_list[0].shape, (n_samples, 2))

    def test_multilabel_predict_proba_list_single_observed_class(self):
        y = np.array(
            [
                [0, 0],
                [0, 0],
                [-1, -1],
                [-1, -1],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            proba_format="list",
        )
        clf.fit(self.X_ml, y)

        P_list = clf.predict_proba(self.X_ml)

        self.assertEqual(len(P_list), 2)
        for P_j in P_list:
            self.assertEqual(P_j.shape, (len(self.X_ml), 2))
            np.testing.assert_array_equal(P_j[:, 1], np.zeros(len(self.X_ml)))

    def test_proba_format_resolution(self):
        clf = self._prefit_multilabel_clf(classes=[[0, 1, 2], [0, 1]])
        clf.proba_format = "invalid"
        self.assertRaises(ValueError, clf._resolve_proba_format)

        clf.proba_format = "auto"
        self.assertEqual(clf._resolve_proba_format(), "list")

        clf.proba_format = "array"
        self.assertRaises(ValueError, clf._resolve_proba_format)

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

    def test_fit_param_sample_weight(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.ones(len(self.fit_default_params["y"]) + 1), ValueError),
        ]
        super().test_fit_param_sample_weight(test_cases=test_cases)

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
            self.X_ml = np.array(
                [[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32
            )
            self.y_ml = np.array(
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                dtype=np.float32,
            )
            init_default_params_multilabel = {
                "classes": [[0, 1], [0, 1]],
                "missing_label": -1,
            }
            fit_default_params_multilabel = {
                "X": self.X_ml,
                "y": self.y_ml,
            }
            pred_def_params_multilabel = {"X": self.X_ml}
            super().setUp(
                estimator_class=estimator_class,
                init_default_params=init_default_params,
                fit_default_params=fit_default_params,
                predict_default_params=predict_default_params,
                init_default_params_multilabel=init_default_params_multilabel,
                fit_default_params_multilabel=fit_default_params_multilabel,
                predict_default_params_multilabel=pred_def_params_multilabel,
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
                (None, None),
                (nn.NLLLoss, None),
                (nn.CrossEntropyLoss, None),
                (nn.NLLLoss(), None),
                (nn.CrossEntropyLoss(), None),
            ]
            self._test_param("init", "criterion", test_cases)

        def test_init_param_target_dtype(self, test_cases=None):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(None, None), (np.int64, None)]
            self._test_param("init", "target_dtype", test_cases)

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
            # Prediction without fit and without classes is ambiguous.
            clf = SkorchClassifier(**self.init_default_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            self.assertRaises(ValueError, clf.predict, self.X)

            # The ambiguity remains even after manual initialization.
            clf = SkorchClassifier(**self.init_default_params)
            clf.initialize()
            self.assertRaises(ValueError, clf.predict, self.X)

            # Providing flat classes disambiguates multiclass prediction.
            init_default_params = self.init_default_params.copy()
            init_default_params["classes"] = [0, 1]
            clf = SkorchClassifier(**init_default_params)
            clf.initialize()
            P = clf.predict_proba(self.X)
            self.assertEqual(P.shape, (len(self.X), 2))
            y_pred = clf.predict(self.X)
            self.assertTrue((np.isin(y_pred, [0, 1])).all())

            # Providing nested classes disambiguates multilabel prediction.
            init_default_params = self.init_default_params.copy()
            init_default_params.update(self.init_default_params_multilabel)
            clf = SkorchClassifier(**init_default_params)
            clf.initialize()
            P = clf.predict_proba(self.X_ml)
            self.assertEqual(P.shape, self.y_ml.shape)
            y_pred = clf.predict(self.X_ml)
            self.assertEqual(y_pred.shape, self.y_ml.shape)

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

            clf = SkorchClassifier(**self.init_default_params)
            self.assertRaises(ValueError, clf.predict_proba, self.X)

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

        def test_multilabel_defaults(self):
            init_params = self.init_default_params.copy()
            init_params.update(self.init_default_params_multilabel)
            clf = SkorchClassifier(**init_params)
            clf.fit(self.X_ml, self.y_ml)
            self.assertIsInstance(
                clf.neural_net_.criterion_, nn.BCEWithLogitsLoss
            )
            P = clf.predict_proba(self.X_ml)
            self.assertEqual(P.shape, self.y_ml.shape)

        def test_skorch_helper_defaults(self):
            init_params = self.init_default_params.copy()
            init_params.update(self.init_default_params_multilabel)
            init_params["criterion"] = nn.BCEWithLogitsLoss
            clf = SkorchClassifier(**init_params)
            forward_outputs = clf._effective_forward_outputs()
            self.assertIn("proba", forward_outputs)
            self.assertIs(forward_outputs["proba"][1], torch.sigmoid)

            clf_no_classes = SkorchClassifier(**self.init_default_params)
            clf_no_classes._initialize_fallbacks(np.zeros((2, 3)))
            np.testing.assert_array_equal(
                clf_no_classes.classes_, np.arange(3)
            )
            self.assertFalse(clf_no_classes.multioutput_)

            clf_multilabel = SkorchClassifier(**init_params)
            clf_multilabel._initialize_fallbacks(np.zeros((2, 2)))
            self.assertTrue(clf_multilabel.multioutput_)
            np.testing.assert_array_equal(
                clf_multilabel.classes_[0], np.array([0, 1])
            )
            np.testing.assert_array_equal(
                clf_multilabel.classes_[1], np.array([0, 1])
            )
            self.assertIsNone(clf_multilabel.cost_matrix_)

            class DummyLoss(nn.Module):
                def forward(self, input, target):
                    return input.sum()

            self.assertEqual(
                clf._infer_target_numpy_dtype(DummyLoss()),
                np.int64,
            )

        def test_prefit_prediction_ambiguity_helper(self):
            clf = SkorchClassifier(**self.init_default_params)
            self.assertRaises(
                ValueError, clf._check_prefit_prediction_ambiguity
            )

            init_params = self.init_default_params.copy()
            init_params["classes"] = [0, 1]
            clf = SkorchClassifier(**init_params)
            clf._check_prefit_prediction_ambiguity()

            clf = SkorchClassifier(**self.init_default_params)
            clf._initialize_fallbacks(np.zeros((2, 3)))
            clf._check_prefit_prediction_ambiguity()

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


if successful_river_import:

    class TestRiverClassifier(TemplateSkactivemlClassifier, unittest.TestCase):
        def setUp(self):
            # Set global seeds.
            random.seed(0)
            self.X, self.y_true = make_blobs(
                n_samples=200, n_features=1, centers=2, random_state=0
            )
            self.X = self.X.astype(np.float32)
            self.y = np.copy(self.y_true).astype(np.float32)
            self.y[:100] = MISSING_LABEL
            self.y_ulbld = np.full_like(self.y, fill_value=MISSING_LABEL)
            self.classes = np.unique(self.y_true)

            estimator_class = RiverClassifier
            init_default_params = {
                "estimator": river.tree.HoeffdingAdaptiveTreeClassifier(
                    seed=0
                ),
                "classes": None,
                "missing_label": MISSING_LABEL,
                "cost_matrix": None,
                "random_state": 0,
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

        def test_init_param_estimator(self):
            test_cases = [
                (Perceptron(), TypeError),
                ("Test", TypeError),
                (GaussianNB(), TypeError),
                (LinearRegression(), TypeError),
                (river.tree.HoeffdingAdaptiveTreeClassifier, TypeError),
                (river.tree.HoeffdingAdaptiveTreeClassifier(), None),
                (river.tree.ExtremelyFastDecisionTreeClassifier(), None),
                (river.tree.LASTClassifier(), None),
                (river.tree.SGTClassifier(), None),
                (river.neighbors.KNNClassifier(), None),
                (river.naive_bayes.MultinomialNB(), None),
                (
                    river.multiclass.OneVsOneClassifier(
                        river.neighbors.KNNClassifier()
                    ),
                    None,
                ),
                (river.forest.AMFClassifier(), None),
                (river.linear_model.LogisticRegression(), None),
            ]
            self._test_param("init", "estimator", test_cases)

        def _test_fit(self, fit_function):
            river_clf = river.multiclass.OneVsRestClassifier(
                river.linear_model.LogisticRegression(),
            )
            for classes_type in ["int", "str"]:
                for provide_classes in [True, False]:
                    subtest_msg = (
                        f"classes_type: {classes_type}, "
                        f"provide_classes: {provide_classes}"
                    )
                    with self.subTest(msg=subtest_msg):
                        if classes_type == "int":
                            classes = [0, 1, 2]
                            missing_label = MISSING_LABEL
                        else:
                            classes = ["0", "1", "2"]
                            missing_label = "unlabeled"
                        if not provide_classes:
                            classes = None
                        clf = RiverClassifier(
                            river_clf,
                            random_state=0,
                            missing_label=missing_label,
                            classes=classes,
                        )
                        fit_func = clf.fit
                        if fit_function == "partial_fit":
                            fit_func = clf.partial_fit
                        X, y_centers = make_blobs(centers=5, random_state=1)
                        y_true = y_centers % 3
                        if classes_type == "str":
                            y_true = y_true.astype(str)
                        y_all_missing = np.full(y_true.shape, missing_label)
                        # check if regular fit was succesful with
                        # is_fitted_=True
                        fit_func(X, y_true)
                        self.assertTrue(clf.is_fitted_)
                        # check that training with empty arrays works with
                        # is_fitted_=False if classes is provided, else a
                        # ValueError is expected
                        if provide_classes:
                            fit_func(X[:0], y_true[:0])
                            self.assertFalse(clf.is_fitted_)
                        else:
                            self.assertRaises(
                                ValueError, fit_func, X[:0], y_true[:0]
                            )
                        # check that training with fully unlabeled data works
                        # with is_fitted_=False if classes is provided, else a
                        # ValueError is expected
                        if provide_classes:
                            fit_func(X, y_all_missing)
                            self.assertFalse(clf.is_fitted_)
                        else:
                            self.assertRaises(
                                ValueError, fit_func, X, y_all_missing
                            )

        def test_fit(self):
            self._test_fit("fit")

        def test_partial_fit(self):
            self._test_fit("partial_fit")
            clfs = {
                "learn_one clf": river.tree.HoeffdingAdaptiveTreeClassifier(
                    seed=0
                ),
                "learn_many clf": river.multiclass.OneVsRestClassifier(
                    river.linear_model.LogisticRegression()
                ),
            }
            for clf_name, river_clf in clfs.items():
                with self.subTest(clf_name):
                    fit_results = {}
                    n_classes = 5
                    # shift classes by 1 to check correct assignment
                    classes = list(range(n_classes + 1))
                    X, y = make_blobs(
                        n_samples=200,
                        centers=n_classes,
                        shuffle=True,
                        random_state=0,
                        cluster_std=1.0,
                    )
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, random_state=0
                    )
                    # check case where not all classes exist
                    # 1-5 exist while there is no instance with y=0
                    # at least predictions should be similar or the same while
                    # predict_proba may differ when learn_many is used
                    y_train += 1
                    y_test += 1
                    for fit_func in ["fit", "partial_fit"]:
                        init_default_params = {
                            "estimator": deepcopy(river_clf),
                            "classes": classes,
                            "missing_label": MISSING_LABEL,
                            "cost_matrix": None,
                            "random_state": 0,
                        }
                        clf = RiverClassifier(**init_default_params)
                        if fit_func == "fit":
                            clf.fit(X_train, y_train)
                        elif fit_func == "partial_fit":
                            for x_inst, y_inst in zip(X_train, y_train):
                                clf.partial_fit([x_inst], [y_inst])
                        pred_result = clf.predict(X_test)
                        pred_proba_result = clf.predict_proba(X_test)
                        results = {}
                        results["predict"] = pred_result
                        results["predict_proba"] = pred_proba_result
                        fit_results[fit_func] = results
                        self.assertTrue(clf.is_fitted_)
                        self.assertAlmostEqual(
                            0, results["predict_proba"][:, 0].sum()
                        )
                    if clf_name == "learn_one clf":
                        np.testing.assert_equal(
                            fit_results["fit"]["predict"],
                            fit_results["partial_fit"]["predict"],
                        )
                        np.testing.assert_almost_equal(
                            fit_results["fit"]["predict_proba"],
                            fit_results["partial_fit"]["predict_proba"],
                        )

        def test_predict(self):
            clfs = {
                "HoeffdingAdaptiveTreeClassifier": (
                    river.tree.HoeffdingAdaptiveTreeClassifier(seed=0)
                ),
                "GaussianNB": river.naive_bayes.GaussianNB(),
            }
            for clf_name, river_clf in clfs.items():
                n_classes = 2
                classes = list(range(n_classes))
                X, y = make_blobs(
                    n_samples=2000,
                    centers=n_classes,
                    shuffle=True,
                    random_state=0,
                    cluster_std=0.01,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=0
                )
                init_default_params = {
                    "estimator": deepcopy(river_clf),
                    "classes": classes,
                    "missing_label": MISSING_LABEL,
                    "cost_matrix": None,
                    "random_state": 0,
                }
                clf = RiverClassifier(**init_default_params)
                clf.fit(X_train, y_train)
                self.assertTrue(clf.is_fitted_)

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    y = y_train
                    if X_str == "X_test":
                        X = X_test
                        y = y_test
                    with self.subTest(f"clf:{clf_name}, X:{X_str}"):
                        pred = clf.predict(X)
                        self.assertEqual(len(pred), len(X))
                        # np.testing.assert_equal(np.unique(pred), classes)
                        # Check that the model learns the classification even
                        # though it might not be perfect
                        accuracy = np.mean(pred == y)
                        self.assertGreaterEqual(accuracy, 0.80)

                clf = RiverClassifier(**init_default_params)
                clf.fit(X_train, np.full(y_train.shape, MISSING_LABEL))

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"no labels, clf:{clf_name}, X:{X_str}"):
                        pred = clf.predict(X)
                        self.assertEqual(len(pred), len(X))
                        self.assertGreater(np.sum(pred == 0), 0)

                clf = RiverClassifier(**init_default_params)
                sample_weight = np.full(y_train.shape, 1.0)
                sample_weight[y_train == 5] = 0.000001
                if clf_name == "GaussianNB":
                    # GaussianNB does not support sample weight
                    self.assertRaises(
                        ValueError, clf.fit, X_train, y_train, sample_weight
                    )
                else:
                    clf.fit(X_train, y_train, sample_weight)
                    self.assertTrue(clf.is_fitted_)

                    for X_str in ["X_train", "X_test"]:
                        X = X_train
                        if X_str == "X_test":
                            X = X_test
                        with self.subTest(
                            f"sample_weight, clf:{clf_name}, X:{X_str}"
                        ):
                            pred = clf.predict(X)
                            self.assertEqual(len(pred), len(X))
                            self.assertEqual(np.sum(pred == 5), 0)

            init_default_params = {
                "estimator": river.linear_model.LogisticRegression(),
                "classes": [0, 1],
                "missing_label": MISSING_LABEL,
                "cost_matrix": None,
                "random_state": 0,
            }
            clf = RiverClassifier(**init_default_params)

            X, y = make_blobs(
                n_samples=200,
                centers=2,
                shuffle=True,
                random_state=0,
                cluster_std=1.0,
            )
            sample_weight = np.full(y.shape, 1.0)
            sample_weight[y == 1] = 0
            clf.fit(X, y, sample_weight)
            pred1 = clf.predict(X)
            clf.fit(X, y)
            pred2 = clf.predict(X)
            self.assertEqual(len(pred1), len(X))
            # check that the number of y=1 predictions decreases
            self.assertGreater(np.sum(pred2 == 1), np.sum(pred1 == 1))

        def test_predict_proba(self):
            clfs = {
                "HoeffdingAdaptiveTreeClassifier": (
                    river.tree.HoeffdingAdaptiveTreeClassifier(seed=0)
                ),
                "GaussianNB": river.naive_bayes.GaussianNB(),
            }
            for clf_name, river_clf in clfs.items():
                n_classes = 5
                classes = list(range(n_classes))
                X, y = make_blobs(
                    n_samples=200,
                    centers=5,
                    shuffle=True,
                    random_state=0,
                    cluster_std=1.0,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=0
                )
                init_default_params = {
                    "estimator": deepcopy(river_clf),
                    "classes": classes,
                    "missing_label": MISSING_LABEL,
                    "cost_matrix": None,
                    "random_state": 0,
                }
                clf = RiverClassifier(**init_default_params)
                clf.fit(X_train, y_train)

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"clf:{clf_name}, X:{X_str}"):
                        pred_proba = clf.predict_proba(X)
                        self.assertEqual(pred_proba.shape[0], len(X))
                        self.assertEqual(pred_proba.shape[1], n_classes)

                clf = RiverClassifier(**init_default_params)
                clf.fit(X_train, np.full(y_train.shape, MISSING_LABEL))

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"no labels, clf:{clf_name}, X:{X_str}"):
                        pred_proba = clf.predict_proba(X)
                        self.assertEqual(pred_proba.shape[0], len(X))
                        self.assertEqual(pred_proba.shape[1], n_classes)
                        np.testing.assert_almost_equal(
                            pred_proba,
                            np.full(pred_proba.shape, 1.0 / n_classes),
                        )

                clf = RiverClassifier(**init_default_params)
                clf.fit(X_train[:, :, None], np.full(y_train.shape, 0))

                # fail training but provide valid labels
                if clf_name == "HoeffdingAdaptiveTreeClassifier":
                    for X_str in ["X_train", "X_test"]:
                        X = X_train
                        if X_str == "X_test":
                            X = X_test
                        with self.subTest(
                            f"no labels, clf:{clf_name}, X:{X_str}"
                        ):
                            pred_proba = clf.predict_proba(X)
                            self.assertEqual(pred_proba.shape[0], len(X))
                            self.assertEqual(pred_proba.shape[1], n_classes)
                            # every predictions should be the same
                            self.assertEqual(pred_proba[:, 0].sum(), len(X))
                            self.assertFalse(clf.is_fitted_)

        def test_is_fitted(self):
            init_params = deepcopy(self.init_default_params)
            init_params["classes"] = [0, 1]
            clf = RiverClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf = RiverClassifier(**init_params)
            clf.fit(**self.fit_default_params)
            check_is_fitted(clf)
            clf = RiverClassifier(**init_params)
            clf.fit(self.X, self.y_ulbld)
            check_is_fitted(clf)

        def test_consistency(self):
            # check if predictions of wrapper are (almost) equal to the actual
            # predictions of the raw river classifier
            clfs = {
                "HoeffdingAdaptiveTreeClassifier": (
                    river.tree.HoeffdingAdaptiveTreeClassifier(seed=0)
                ),
                "GaussianNB": river.naive_bayes.GaussianNB(),
                "LR": river.linear_model.LogisticRegression(),
            }
            for clf_name, river_clf in clfs.items():
                n_classes = 2
                classes = list(range(n_classes))
                X, y = make_blobs(
                    n_samples=200,
                    centers=n_classes,
                    shuffle=True,
                    random_state=0,
                    cluster_std=1.0,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=0
                )
                init_default_params = {
                    "estimator": deepcopy(river_clf),
                    "classes": classes,
                    "missing_label": MISSING_LABEL,
                    "cost_matrix": None,
                    "random_state": 0,
                }
                clf = RiverClassifier(**init_default_params)
                clf.fit(X_train, y_train)
                pred_proba = clf.predict_proba(X_test)

                pred_proba_river = self.evaluate_river_model(
                    deepcopy(river_clf), X_train, y_train, X_test, clf.classes_
                )
                np.testing.assert_almost_equal(pred_proba, pred_proba_river)

        def evaluate_river_model(
            self, base_clf, X_train, y_train, X_test, classes
        ):
            # fit a river classifier and returns its predict_proba results of
            # from river classifiers
            X_train_pd = self.prepare_river_data(X_train)
            y_train_pd = pd.Series(y_train)
            X_test_pd = self.prepare_river_data(X_test)
            river_clf = deepcopy(base_clf)
            if hasattr(river_clf, "learn_many"):
                river_clf.learn_many(X_train_pd, y_train_pd)
            else:
                for i in range(len(X_train)):
                    X_train_pd_i = X_train_pd.iloc[i]
                    y_train_pd_i = y_train_pd.iloc[i]
                    river_clf.learn_one(X_train_pd_i, y_train_pd_i)
            pred_proba = []
            for i, X_test_pd_i in X_test_pd.iterrows():
                pred_proba_river = river_clf.predict_proba_one(
                    X_test_pd_i.to_dict()
                )
                sorted_pred_proba_entry = []
                for c in classes:
                    sorted_pred_proba_entry.append(pred_proba_river[c])
                pred_proba.append(sorted_pred_proba_entry)
            return np.array(pred_proba)

        def prepare_river_data(self, data):
            column_names = [f"X{f_i}" for f_i in range(data.shape[1])]
            data_dict = {
                col_name: col for col_name, col in zip(column_names, data.T)
            }
            return pd.DataFrame(data_dict)


if successful_capymoa_import:
    from skactiveml.classifier import CapyMOAClassifier

    class TestCapyMOAClassifier(
        TemplateSkactivemlClassifier, unittest.TestCase
    ):
        def setUp(self):
            from capymoa.classifier import AdaptiveRandomForestClassifier

            # Set global seeds.
            random.seed(0)
            self.X, self.y_true = make_blobs(
                n_samples=200, n_features=1, centers=2, random_state=0
            )
            self.X = self.X.astype(np.float32)
            self.y = np.copy(self.y_true).astype(np.float32)
            self.y[:100] = MISSING_LABEL
            self.y_ulbld = np.full_like(self.y, fill_value=MISSING_LABEL)
            self.classes = np.unique(self.y_true)

            estimator_class = CapyMOAClassifier
            init_default_params = {
                "estimator_class": AdaptiveRandomForestClassifier,
                "classes": None,
                "missing_label": MISSING_LABEL,
                "cost_matrix": None,
                "random_state": 0,
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

        def test_init_param_estimator_class(self):
            from capymoa.classifier import AdaptiveRandomForestClassifier
            from capymoa.stream import Schema

            schema = Schema.from_custom(
                ["f0", "target", "f1"],
                target="target",
                categories={"target": ["0", "1", "2", "3"]},
                name="test_ds",
            )
            test_cases = [
                (Perceptron(), TypeError),
                ("Test", TypeError),
                (GaussianNB(), TypeError),
                (LinearRegression(), TypeError),
                (AdaptiveRandomForestClassifier(schema=schema), TypeError),
                (AdaptiveRandomForestClassifier, None),
            ]
            self._test_param("init", "estimator_class", test_cases)

        def test_init_param_estimator_param_dict(self):
            from capymoa.stream import Schema

            schema = Schema.from_custom(
                ["f0", "target", "f1"],
                target="target",
                categories={"target": ["0", "1", "2", "3"]},
                name="test_ds",
            )
            test_cases = [
                ("Test", TypeError),
                ([("disable_drift_detection", True)], TypeError),
                (
                    {"disable_drift_detection": True, "schema": schema},
                    AttributeError,
                ),
                ({"disable_drift_detection": True}, None),
            ]
            self._test_param("init", "estimator_param_dict", test_cases)

        def _test_fit(self, fit_function):
            from capymoa.classifier import AdaptiveRandomForestClassifier

            capymoa_clf_class = AdaptiveRandomForestClassifier
            for classes_type in ["int", "str"]:
                for provide_classes in [True, False]:
                    subtest_msg = (
                        f"classes_type: {classes_type}, "
                        f"provide_classes: {provide_classes}"
                    )
                    with self.subTest(msg=subtest_msg):
                        if classes_type == "int":
                            classes = [0, 1, 2]
                            missing_label = MISSING_LABEL
                        else:
                            classes = ["0", "1", "2"]
                            missing_label = "unlabeled"
                        if not provide_classes:
                            classes = None
                        clf = CapyMOAClassifier(
                            estimator_class=capymoa_clf_class,
                            random_state=0,
                            missing_label=missing_label,
                            classes=classes,
                        )
                        fit_func = clf.fit
                        if fit_function == "partial_fit":
                            fit_func = clf.partial_fit
                        X, y_centers = make_blobs(centers=5, random_state=1)
                        y_true = y_centers % 3
                        if classes_type == "str":
                            y_true = y_true.astype(str)
                        y_all_missing = np.full(y_true.shape, missing_label)
                        # check if regular fit was succesful with
                        # is_fitted_=True
                        fit_func(X, y_true)
                        self.assertTrue(clf.is_fitted_)
                        # check that training with empty arrays works with
                        # is_fitted_=False if classes is provided, else a
                        # ValueError is expected
                        if provide_classes:
                            fit_func(X[:0], y_true[:0])
                            self.assertFalse(clf.is_fitted_)
                        else:
                            self.assertRaises(
                                ValueError, fit_func, X[:0], y_true[:0]
                            )
                        # check that training with fully unlabeled data works
                        # with is_fitted_=False if classes is provided, else a
                        # ValueError is expected
                        if provide_classes:
                            fit_func(X, y_all_missing)
                            self.assertFalse(clf.is_fitted_)
                        else:
                            self.assertRaises(
                                ValueError, fit_func, X, y_all_missing
                            )

        def test_fit(self):
            self._test_fit("fit")

        def test_partial_fit(self):
            self._test_fit("partial_fit")

        def test_predict(self):
            from capymoa.classifier import (
                AdaptiveRandomForestClassifier,
                DynamicWeightedMajority,
            )

            estimator_classes = {
                "AdaptiveRandomForestClassifier": (
                    AdaptiveRandomForestClassifier
                ),
                "DynamicWeightedMajority": DynamicWeightedMajority,
            }
            for est_name, est_cls in estimator_classes.items():
                n_classes = 10
                classes = list(range(n_classes))
                X, y = make_blobs(
                    n_samples=200,
                    centers=n_classes,
                    shuffle=True,
                    random_state=0,
                    cluster_std=1.0,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=0
                )
                init_default_params = {
                    "estimator_class": est_cls,
                    "classes": classes,
                    "missing_label": MISSING_LABEL,
                    "cost_matrix": None,
                    "random_state": 0,
                }
                clf = CapyMOAClassifier(**init_default_params)
                clf.fit(X_train, y_train)

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    y = y_train
                    if X_str == "X_test":
                        X = X_test
                        y = y_test
                    with self.subTest(f"clf:{est_name}, X:{X_str}"):
                        pred = clf.predict(X)
                        self.assertEqual(len(pred), len(X))
                        np.testing.assert_equal(np.unique(pred), classes)
                        # Check that the model learns the classification even
                        # though it might not be perfect
                        accuracy = np.mean(pred == y)
                        self.assertGreaterEqual(accuracy, 0.80)

                clf = CapyMOAClassifier(**init_default_params)
                clf.fit(X_train, np.full(y_train.shape, MISSING_LABEL))

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"no labels, clf:{est_name}, X:{X_str}"):
                        pred = clf.predict(X)
                        self.assertEqual(len(pred), len(X))
                        self.assertGreater(np.sum(pred == 0), 0)

        def test_predict_proba(self):
            from capymoa.classifier import (
                AdaptiveRandomForestClassifier,
                DynamicWeightedMajority,
            )

            estimator_classes = {
                "AdaptiveRandomForestClassifier": (
                    AdaptiveRandomForestClassifier
                ),
                "DynamicWeightedMajority": DynamicWeightedMajority,
            }
            for est_name, est_cls in estimator_classes.items():
                n_classes = 5
                classes = list(range(n_classes))
                X, y = make_blobs(
                    n_samples=200,
                    centers=5,
                    shuffle=True,
                    random_state=0,
                    cluster_std=1.0,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=0
                )
                init_default_params = {
                    "estimator_class": deepcopy(est_cls),
                    "classes": classes,
                    "missing_label": MISSING_LABEL,
                    "cost_matrix": None,
                    "random_state": 0,
                }
                clf = CapyMOAClassifier(**init_default_params)
                clf.fit(X_train, y_train)

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"clf:{est_name}, X:{X_str}"):
                        pred_proba = clf.predict_proba(X)
                        self.assertEqual(pred_proba.shape[0], len(X))
                        self.assertEqual(pred_proba.shape[1], n_classes)

                clf = CapyMOAClassifier(**init_default_params)
                clf.fit(X_train, np.full(y_train.shape, MISSING_LABEL))

                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"no labels, clf:{est_name}, X:{X_str}"):
                        pred_proba = clf.predict_proba(X)
                        self.assertEqual(pred_proba.shape[0], len(X))
                        self.assertEqual(pred_proba.shape[1], n_classes)
                        np.testing.assert_almost_equal(
                            pred_proba,
                            np.full(pred_proba.shape, 1.0 / n_classes),
                        )

                clf = CapyMOAClassifier(**init_default_params)
                clf.fit(X_train[:, :, None], np.full(y_train.shape, 0))

                # fail training but provide valid labels
                for X_str in ["X_train", "X_test"]:
                    X = X_train
                    if X_str == "X_test":
                        X = X_test
                    with self.subTest(f"no labels, clf:{est_name}, X:{X_str}"):
                        pred_proba = clf.predict_proba(X)
                        self.assertEqual(pred_proba.shape[0], len(X))
                        self.assertEqual(pred_proba.shape[1], n_classes)
                        # every predictions should be the same
                        self.assertEqual(pred_proba[:, 0].sum(), len(X))
                        self.assertFalse(clf.is_fitted_)

        def test_is_fitted(self):
            init_params = deepcopy(self.init_default_params)
            init_params["classes"] = [0, 1]
            clf = CapyMOAClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf = CapyMOAClassifier(**init_params)
            clf.fit(**self.fit_default_params)
            check_is_fitted(clf)
            clf = CapyMOAClassifier(**init_params)
            clf.fit(self.X, self.y_ulbld)
            check_is_fitted(clf)

        def test_consistency(self):
            # check if predictions of wrapper are (almost) equal to the actual
            # predictions of the raw river classifier
            from capymoa.classifier import (
                AdaptiveRandomForestClassifier,
                DynamicWeightedMajority,
            )

            clfs = {
                "AdaptiveRandomForestClassifier": (
                    AdaptiveRandomForestClassifier
                ),
                "DynamicWeightedMajority": DynamicWeightedMajority,
            }
            for clf_name, estimator_class in clfs.items():
                n_classes = 2
                classes = list(range(n_classes))
                X, y = make_blobs(
                    n_samples=200,
                    centers=n_classes,
                    shuffle=True,
                    random_state=0,
                    cluster_std=1.0,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=0
                )
                init_default_params = {
                    "estimator_class": estimator_class,
                    "classes": classes,
                    "missing_label": MISSING_LABEL,
                    "cost_matrix": None,
                    "random_state": 0,
                }
                clf = CapyMOAClassifier(**init_default_params)
                clf.fit(X_train, y_train)
                pred_proba = clf.predict_proba(X_test)

                pred_proba_river = self.evaluate_capymoa_model(
                    estimator_class, X_train, y_train, X_test, clf.classes_
                )
                np.testing.assert_almost_equal(pred_proba, pred_proba_river)

        def evaluate_capymoa_model(
            self, base_clf_cls, X_train, y_train, X_test, classes
        ):
            # fit a capymoa classifier and returns its predict_proba results of
            # from capymoa classifiers
            from capymoa.stream import Schema
            from capymoa.instance import LabeledInstance, Instance

            column_list = [f"f{i}" for i in range(X_train.shape[1])]
            column_list += ["label"]
            schema = Schema.from_custom(
                features=column_list,
                target="label",
                categories={"label": classes},
            )
            clf = base_clf_cls(schema=schema)
            # we assume that classes is from 0 to C
            for i in range(len(X_train)):
                instance = LabeledInstance.from_array(
                    schema=schema,
                    x=X_train[i],
                    y_index=y_train[i],
                )
                clf.train(instance)
            pred_proba = []
            for i in range(len(X_test)):
                instance = Instance.from_array(
                    schema=schema, instance=X_test[i]
                )
                pred_proba_capymoa = clf.predict_proba(instance)
                if pred_proba_capymoa is None:
                    pred_proba_capymoa = np.ones(
                        shape=(classes,), dtype=float
                    ) / len(classes)
                pred_proba.append(pred_proba_capymoa)
            return np.array(pred_proba)
