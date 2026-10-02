import random
import unittest
import warnings
import inspect

from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.dummy import DummyClassifier
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
from sklearn.utils import get_tags
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
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
from skactiveml.tests.utils import (
    assert_attributes_unchanged,
    assert_fit_failure_is_transactional,
    assert_predicts_class_dtype,
)
from skactiveml.utils import MISSING_LABEL, TargetSpec

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
    non_integral_classes_error = RuntimeError

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
        def __init__(
            self, proba, classes_=None, estimators_=None, predictions=None
        ):
            self._proba = proba
            self._predictions = predictions
            if classes_ is not None:
                self.classes_ = classes_
            if estimators_ is not None:
                self.estimators_ = estimators_

        def predict_proba(self, X, **kwargs):
            return self._proba

        def predict(self, X, **kwargs):
            if self._predictions is not None:
                return self._predictions
            return np.zeros((len(X),), dtype=int)

    class _TwoOutputEstimator(ClassifierMixin, BaseEstimator):
        """Classifier for two binary outputs, declaring no capability tag."""

        def fit(self, X, y):
            self.X_fit_ = np.asarray(X).copy()
            self.y_fit_ = np.asarray(y).copy()
            self.classes_ = [np.array([0, 1]), np.array([0, 1])]
            return self

        def predict(self, X):
            return np.zeros((len(X), 2), dtype=int)

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    class _ReorderedClassesEstimator(ClassifierMixin, BaseEstimator):
        """Classifier publishing its learned classes in unsorted order."""

        def fit(self, X, y):
            self.classes_ = np.array([1, 0])
            self.n_features_in_ = np.shape(X)[1]
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            # `0.8` is the probability of class `1` and `0.2` of class `0`.
            return np.tile([0.8, 0.2], (len(X), 1))

    class _NoClassesEstimator(ClassifierMixin, BaseEstimator):
        """Fitted classifier publishing no learned class vocabulary."""

        def fit(self, X, y):
            self.n_features_in_ = np.shape(X)[1]
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

    class _TargetSpecEstimator(ClassifierMixin, BaseEstimator):
        """Fitted classifier publishing a target specification of its own."""

        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = np.shape(X)[1]
            self.target_spec_ = TargetSpec(
                task="classification",
                target_type="single-output",
                annotation_type="single-annotator",
                classes=(0, 1),
            )
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

    class _MultiOutputTaggedEstimator(_TwoOutputEstimator):
        """Two-output classifier declaring `target_tags.multi_output`."""

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.multi_output = True
            return tags

    class _NaNClassMultiOutputEstimator(_MultiOutputTaggedEstimator):
        """Classifier learning a NaN class in a non-canonical order."""

        def fit(self, X, y):
            self.classes_ = [
                np.array([np.nan, 1.0]),
                np.array([1.0, 0.0]),
            ]
            self.n_features_in_ = np.shape(X)[1]
            return self

        def predict(self, X):
            return np.tile([np.nan, 1.0], (len(X), 1))

        def predict_proba(self, X):
            return [
                np.tile([0.75, 0.25], (len(X), 1)),
                np.tile([0.6, 0.4], (len(X), 1)),
            ]

    class _MultiLabelTaggedEstimator(_TwoOutputEstimator):
        """Two-output classifier declaring only `classifier_tags.multi_label`.

        This isolates the second admission tag, which the `scikit-learn`
        estimators declaring `target_tags.multi_output` cannot exercise.
        """

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.classifier_tags.multi_label = True
            return tags

    class _NestedListProbaEstimator(_MultiOutputTaggedEstimator):
        """Classifier returning positive-class probabilities as Python rows."""

        def predict_proba(self, X):
            probabilities = [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ]
            return probabilities[: len(X)]

    class _CallRecordingEstimator(_MultiOutputTaggedEstimator):
        """Admitted classifier recording every fit and inference call."""

        def __init__(self):
            self.calls = []

        def fit(self, X, y):
            self.calls.append("fit")
            return super().fit(X, y)

        def predict(self, X):
            self.calls.append("predict")
            return super().predict(X)

        def predict_proba(self, X):
            self.calls.append("predict_proba")
            return super().predict_proba(X)

    class _BrokenEstimator(_MultiOutputTaggedEstimator):
        """Admitted classifier whose fit always fails unexpectedly."""

        def fit(self, X, y):
            raise RuntimeError("the estimator is broken")

    @staticmethod
    def _prefit_multilabel_clf(proba_format="array", classes=None):
        if classes is None:
            classes = [[0, 1], [0, 1]]
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=classes,
            missing_label=-1,
            proba_format=proba_format,
            random_state=0,
        )
        clf.check_X_dict_ = {"ensure_min_samples": 0, "ensure_min_features": 0}
        clf.n_features_in_ = 1
        dummy_classes = np.array([[classes[0][0], classes[1][0]]], dtype=int)
        clf._commit_label_state(clf._resolve_label_state(dummy_classes))
        clf.is_fitted_ = True
        return clf

    def _fit_nan_class_multilabel_clf(self):
        y = np.array(
            [
                [np.nan, 0.0],
                [1.0, 1.0],
                [np.nan, 1.0],
                [1.0, 0.0],
            ]
        )
        return SklearnClassifier(
            estimator=self._NaNClassMultiOutputEstimator(),
            classes=[[np.nan, 1.0], [0.0, 1.0]],
            missing_label=-1,
            proba_format="list",
        ).fit(self.X_ml, y)

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

    def test_partial_fit_reuses_established_target_spec(self):
        clf = SklearnClassifier(estimator=GaussianNB(), missing_label=-1)
        X = np.array([[0.0], [1.0], [2.0], [3.0]])

        clf.partial_fit(X, np.array([0, 1, 0, 1]))
        established_spec = clf.target_spec_

        clf.partial_fit(X[:2], np.array([1, 1]))

        self.assertIs(clf.target_spec_, established_spec)
        self.assertEqual(clf.target_spec_.classes, (0, 1))
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_first_partial_fit_uses_supplied_class_vocabulary(self):
        clf = SklearnClassifier(
            estimator=SGDClassifier(loss="log_loss", random_state=0),
            missing_label=-1,
        )
        X = np.array([[0.0], [1.0]])

        clf.partial_fit(X, np.array([0, 0]), classes=[0, 1])

        self.assertEqual(clf.target_spec_.classes, (0, 1))
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        np.testing.assert_array_equal(clf.estimator_.classes_, [0, 1])
        self.assertTrue(clf.is_fitted_)

    def test_partial_fit_validates_configured_class_vocabulary(self):
        clf = SklearnClassifier(
            estimator=SGDClassifier(loss="log_loss", random_state=0),
            classes=[0, 1],
            missing_label=-1,
        )
        X = np.array([[0.0], [1.0]])

        clf.partial_fit(X, np.array([0, 1]), classes=[0, 1])

        self.assertTrue(clf.is_fitted_)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_partial_fit_rejects_unseen_class_before_mutating_state(self):
        clf = SklearnClassifier(estimator=GaussianNB(), missing_label=-1)
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        clf.partial_fit(X, np.array([0, 1, 0, 1]))
        established_spec = clf.target_spec_
        established_counts = clf.estimator_.class_count_.copy()

        with self.assertRaisesRegex(ValueError, "class"):
            clf.partial_fit(X[:1], np.array([2]))

        self.assertIs(clf.target_spec_, established_spec)
        np.testing.assert_array_equal(
            clf.estimator_.class_count_, established_counts
        )
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_partial_fit_reuses_multilabel_vocabularies(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(
                SGDClassifier(loss="log_loss", random_state=0)
            ),
            missing_label=-1,
            target_type="multi-label",
            proba_format="array",
        )
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        clf.partial_fit(X, y)
        established_spec = clf.target_spec_

        clf.partial_fit(X[:2], np.array([[1, 0], [1, 0]]))

        self.assertIs(clf.target_spec_, established_spec)
        self.assertEqual(clf.target_spec_.classes, ((0, 1), (0, 1)))
        for classes in clf.classes_:
            np.testing.assert_array_equal(classes, [0, 1])

    def test_partial_fit_preserves_native_flat_classes(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]]
        )
        estimators = [
            OneVsRestClassifier(
                SGDClassifier(loss="log_loss", random_state=0)
            ),
            MLPClassifier(max_iter=1, random_state=0),
        ]
        for estimator in estimators:
            for n_outputs in [1, 3]:
                with self.subTest(
                    estimator=type(estimator).__name__, n_outputs=n_outputs
                ):
                    classes = [[0, 1]] * n_outputs
                    y_fit = y[:, :n_outputs]
                    clf = SklearnClassifier(estimator, classes=classes).fit(
                        X, y_fit
                    )
                    established_spec = clf.target_spec_
                    native = deepcopy(clf.estimator_)

                    for kwargs in [{}, {"classes": classes}]:
                        native.partial_fit(X, y_fit)
                        clf.partial_fit(X, y_fit, **kwargs)

                        self.assertIs(clf.target_spec_, established_spec)
                        np.testing.assert_array_equal(
                            clf.estimator_.classes_, native.classes_
                        )
                        np.testing.assert_array_equal(clf.classes_, classes)
                        expected = native.predict_proba(X)
                        if n_outputs == 1:
                            expected = expected[:, [1]]
                        np.testing.assert_allclose(
                            clf.predict_proba(X), expected
                        )

    def test_empty_partial_fit_preserves_fitted_multilabel_state(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(
                SGDClassifier(loss="log_loss", random_state=0)
            ),
            missing_label=-1,
            target_type="multi-label",
            proba_format="array",
        )
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        clf.partial_fit(X, y)
        established_spec = clf.target_spec_
        established_estimator = clf.estimator_
        established_counts = deepcopy(clf._label_counts)
        established_probabilities = clf.predict_proba(X)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clf.partial_fit(np.empty((0, 1)), np.empty((0, 2), dtype=int))

        self.assertTrue(clf.is_fitted_)
        self.assertIs(clf.target_spec_, established_spec)
        self.assertIs(clf.estimator_, established_estimator)
        self.assertEqual(clf.target_spec_.classes, ((0, 1), (0, 1)))
        for actual, expected in zip(clf._label_counts, established_counts):
            np.testing.assert_array_equal(actual, expected)
        for classes in clf.classes_:
            np.testing.assert_array_equal(classes, [0, 1])
        np.testing.assert_allclose(
            clf.predict_proba(X), established_probabilities
        )

    def test_unlabeled_partial_fit_preserves_fitted_state(self):
        clf = SklearnClassifier(
            estimator=GaussianNB(), classes=[0, 1], missing_label=-1
        )
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        clf.partial_fit(X, y)
        established_spec = clf.target_spec_
        established_estimator = clf.estimator_
        established_counts = deepcopy(clf._label_counts)
        established_class_counts = clf.estimator_.class_count_.copy()
        established_probabilities = clf.predict_proba(X)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clf.partial_fit(X[:2], np.full(2, -1))

        self.assertTrue(clf.is_fitted_)
        self.assertIs(clf.target_spec_, established_spec)
        self.assertIs(clf.estimator_, established_estimator)
        np.testing.assert_array_equal(clf._label_counts, established_counts)
        np.testing.assert_array_equal(
            clf.estimator_.class_count_, established_class_counts
        )
        np.testing.assert_allclose(
            clf.predict_proba(X), established_probabilities
        )

    def test_partial_fit_rejects_changed_target_declaration(self):
        clf = SklearnClassifier(estimator=GaussianNB(), missing_label=-1)
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        established_counts = clf.estimator_.class_count_.copy()

        clf.target_type = "multi-label"
        with self.assertRaises(ValueError):
            clf.partial_fit(X, np.array([0, 1]))

        self.assertIs(clf.target_spec_, established_spec)
        np.testing.assert_array_equal(
            clf.estimator_.class_count_, established_counts
        )

    def test_partial_fit_rejects_changed_class_vocabulary(self):
        clf = SklearnClassifier(
            estimator=GaussianNB(), classes=[0, 1], missing_label=-1
        )
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        established_estimator = clf.estimator_

        clf.classes = [0, 2]
        with self.assertRaises(ValueError):
            clf.partial_fit(X[:1], np.array([0]))

        self.assertIs(clf.target_spec_, established_spec)
        self.assertIs(clf.estimator_, established_estimator)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_fit_reinitializes_target_spec_after_partial_fit(self):
        clf = SklearnClassifier(estimator=GaussianNB(), missing_label=-1)
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        clf.partial_fit(X, np.array([0, 1, 0, 1]))
        established_spec = clf.target_spec_

        clf.fit(X, np.array([2, 3, 2, 3]))

        self.assertIsNot(clf.target_spec_, established_spec)
        self.assertEqual(clf.target_spec_.classes, (2, 3))
        np.testing.assert_array_equal(clf.classes_, [2, 3])

    def test_fit_include_unlabeled_samples_with_self_training(self):
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

    def test_multilabel_fit_includes_complete_unlabeled_rows(self):
        y = self.y_ml.copy()
        y[1] = -1
        clf = SklearnClassifier(
            estimator=self._MultiOutputTaggedEstimator(),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            include_unlabeled_samples=True,
        )

        clf.fit(self.X_ml, y)

        self.assertTrue(clf.is_fitted_)
        np.testing.assert_array_equal(clf.estimator_.X_fit_, self.X_ml)
        np.testing.assert_array_equal(clf.estimator_.y_fit_, y)

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

    def test_predict_dtype_with_cost_matrix(self):
        # The cost-matrix path decodes through the label encoder, which
        # `missing_label=np.nan` widens to `float64`. The samples are
        # separable so that the estimator is fitted on real probabilities.
        X = np.array([[-2.0], [0.0], [2.0]])
        y = np.array([0, np.nan, 1])
        clf = SklearnClassifier(
            estimator=GaussianNB(),
            classes=[0, 1],
            missing_label=np.nan,
            cost_matrix=1 - np.eye(2),
        ).fit(X, y)

        y_pred = clf.predict(X)

        self.assertTrue(clf.is_fitted_)
        assert_predicts_class_dtype(self, y_pred, clf.classes_)

    def test_predict_dtype_without_fitted_estimator(self):
        # Without any labels, `predict` falls back to the label prior, which
        # is the path taken in the first cycle of an active learning loop.
        X = np.zeros((3, 1))
        y = np.full(3, np.nan)
        clf = SklearnClassifier(
            estimator=GaussianNB(), classes=[0, 1], missing_label=np.nan
        ).fit(X, y)

        y_pred = clf.predict(X)

        self.assertFalse(clf.is_fitted_)
        assert_predicts_class_dtype(self, y_pred, clf.classes_)

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

    def test_multilabel_predict_accepts_observed_nan_class(self):
        clf = self._fit_nan_class_multilabel_clf()
        predictions = clf.predict(self.X_ml)

        self.assertTrue(np.isnan(predictions[:, 0]).all())
        np.testing.assert_array_equal(predictions[:, 1], 1.0)

    def test_multilabel_predict_proba_maps_observed_nan_class(self):
        clf = self._fit_nan_class_multilabel_clf()
        probabilities = clf.predict_proba(self.X_ml)

        np.testing.assert_allclose(
            probabilities[0],
            np.tile([0.25, 0.75], (len(self.X_ml), 1)),
        )
        np.testing.assert_allclose(
            probabilities[1], np.tile([0.4, 0.6], (len(self.X_ml), 1))
        )

    def test_init_param_target_type(self):
        self._test_param(
            "init",
            "target_type",
            [
                ("auto", None),
                ("single-output", None),
                ("invalid", ValueError),
                (1, ValueError),
                ("multi-label", ValueError),
                ("multi-output", ValueError),
            ],
        )

    def test_explicit_multilabel_without_declared_classes_resolves_on_fit(
        self,
    ):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            missing_label=-1,
            target_type="multi-label",
        )

        clf.fit(self.X_ml, self.y_ml)

        self.assertEqual(clf.target_spec_.target_type, "multi-label")
        self.assertEqual(clf.target_spec_.classes, ((0, 1), (0, 1)))
        for classes_i in clf.classes_:
            np.testing.assert_array_equal(classes_i, [0, 1])

    def test_multilabel_resolution_failure_does_not_commit_fitted_state(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
        )
        y = np.array([[0, 1], [1, np.nan], [np.nan, np.nan]])

        with self.assertRaisesRegex(ValueError, "no mixing within a row"):
            clf.fit(np.arange(3).reshape(-1, 1), y)

        self.assertFalse(hasattr(clf, "target_spec_"))
        self.assertFalse(hasattr(clf, "classes_"))
        self.assertFalse(hasattr(clf, "estimator_"))

    def test_multi_output_capability_failure_does_not_commit_fitted_state(
        self,
    ):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1, 2], [0, 1]],
            target_type="multi-output",
        )
        y = np.array([[0, 1], [1, 0], [2, 1]])

        with self.assertRaisesRegex(ValueError, "does not support"):
            clf.fit(np.arange(3).reshape(-1, 1), y)

        self.assertFalse(hasattr(clf, "target_spec_"))
        self.assertFalse(hasattr(clf, "classes_"))
        self.assertFalse(hasattr(clf, "estimator_"))

    MULTILABEL_CAPABILITY = (
        "classification",
        "multi-label",
        "single-annotator",
    )

    def test_predict_proba_alone_does_not_advertise_multilabel(self):
        clf = SklearnClassifier(estimator=GaussianNB(), missing_label=-1)

        self.assertTrue(hasattr(clf.estimator, "predict_proba"))
        self.assertNotIn(self.MULTILABEL_CAPABILITY, clf._target_capabilities)

    def test_multi_output_tag_advertises_multilabel(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()), missing_label=-1
        )

        self.assertIn(self.MULTILABEL_CAPABILITY, clf._target_capabilities)

    def test_multi_label_tag_alone_advertises_multilabel(self):
        estimator = self._MultiLabelTaggedEstimator()
        clf = SklearnClassifier(estimator=estimator, missing_label=-1)

        self.assertFalse(get_tags(estimator).target_tags.multi_output)
        self.assertTrue(get_tags(estimator).classifier_tags.multi_label)
        self.assertIn(self.MULTILABEL_CAPABILITY, clf._target_capabilities)

    def test_missing_predict_proba_revokes_multilabel_capability(self):
        clf = SklearnClassifier(estimator=Perceptron(), missing_label=-1)

        self.assertNotIn(self.MULTILABEL_CAPABILITY, clf._target_capabilities)

    def test_capability_detection_never_fits_or_predicts(self):
        estimator = self._CallRecordingEstimator()
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
        )

        self.assertIn(self.MULTILABEL_CAPABILITY, clf._target_capabilities)
        clf._resolve_target_spec(self.y_ml)
        self.assertEqual(estimator.calls, [])

    def _assert_multilabel_rejection(self, clf, action):
        attributes_before = dict(clf.__dict__)

        with self.assertRaisesRegex(
            ValueError, "does not support multi-label classification"
        ) as context:
            action()

        self.assertIn("multi_output", str(context.exception))
        self.assertIn("multi_label", str(context.exception))
        assert_attributes_unchanged(self, clf, attributes_before)

    def test_unfitted_multilabel_rejects_undeclared_estimator(self):
        for estimator in [LogisticRegression(), GaussianNB()]:
            with self.subTest(estimator=type(estimator).__name__):
                clf = SklearnClassifier(
                    estimator=estimator,
                    classes=[[0, 1], [0, 1]],
                    missing_label=-1,
                )

                self._assert_multilabel_rejection(
                    clf, lambda: clf.fit(self.X_ml, self.y_ml)
                )

    def test_prefit_multilabel_rejects_undeclared_estimator(self):
        estimator = LogisticRegression().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
        )

        self._assert_multilabel_rejection(clf, lambda: check_is_fitted(clf))

    def _assert_multilabel_contract(self, estimator, proba_format):
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            proba_format=proba_format,
        ).fit(self.X_ml, self.y_ml)

        P = clf.predict_proba(self.X_ml)

        self.assertTrue(clf.is_fitted_)
        self.assertEqual(clf.predict(self.X_ml).shape, self.y_ml.shape)
        if proba_format == "array":
            self.assertEqual(P.shape, self.y_ml.shape)
        else:
            self.assertEqual(len(P), self.y_ml.shape[1])
            for P_j in P:
                self.assertEqual(P_j.shape, (len(self.X_ml), 2))

    def test_tagged_estimators_remain_supported(self):
        # `MultiOutputClassifier` and `RandomForestClassifier` natively return
        # one probability matrix per output, `OneVsRestClassifier` returns one
        # positive-class probability array, and both public `proba_format`
        # values have to be served from either representation.
        estimators = {
            "MultiOutputClassifier": MultiOutputClassifier(GaussianNB()),
            "OneVsRestClassifier": OneVsRestClassifier(LogisticRegression()),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=5, random_state=0
            ),
        }
        for name, estimator in estimators.items():
            for proba_format in ["array", "list"]:
                with self.subTest(estimator=name, proba_format=proba_format):
                    self._assert_multilabel_contract(estimator, proba_format)

    def test_one_output_multilabel_accepts_collapsed_binary_outputs(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        targets = {
            "both-classes": np.array([[0], [0], [1], [1]]),
            "negative-only": np.zeros((len(X), 1), dtype=int),
            "positive-only": np.ones((len(X), 1), dtype=int),
        }
        estimators = {
            "OneVsRestClassifier": OneVsRestClassifier(LogisticRegression()),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=5, random_state=0
            ),
        }

        for name, estimator in estimators.items():
            for target_name, y in targets.items():
                for proba_format in ["array", "list"]:
                    with self.subTest(
                        estimator=name,
                        target=target_name,
                        proba_format=proba_format,
                    ):
                        clf = SklearnClassifier(
                            estimator=estimator,
                            classes=[[0, 1]],
                            missing_label=-1,
                            target_type="multi-label",
                            proba_format=proba_format,
                        ).fit(X, y)

                        predictions = clf.predict(X)
                        probabilities = clf.predict_proba(X)

                        np.testing.assert_array_equal(predictions, y)
                        if proba_format == "array":
                            positive_probabilities = probabilities
                            self.assertEqual(probabilities.shape, y.shape)
                        else:
                            self.assertEqual(len(probabilities), 1)
                            self.assertEqual(
                                probabilities[0].shape, (len(X), 2)
                            )
                            positive_probabilities = probabilities[0][:, [1]]
                        np.testing.assert_array_equal(
                            positive_probabilities >= 0.5, y
                        )

    def test_collapsed_single_class_probabilities_preserve_total_mass(self):
        X = np.arange(4, dtype=float).reshape(-1, 1)
        for observed_class in [0, 1]:
            for proba_format in ["array", "list"]:
                with self.subTest(
                    observed_class=observed_class, proba_format=proba_format
                ):
                    y = np.full((len(X), 1), observed_class)
                    clf = SklearnClassifier(
                        MLPClassifier(max_iter=1, random_state=0),
                        classes=[[0, 1]],
                        proba_format=proba_format,
                    ).fit(X, y)

                    P = clf.predict_proba(X)

                    np.testing.assert_array_equal(clf.predict(X), y)
                    if proba_format == "array":
                        np.testing.assert_allclose(P, y)
                    else:
                        self.assertEqual(len(P), 1)
                        np.testing.assert_allclose(
                            P[0], np.column_stack([1 - y[:, 0], y[:, 0]])
                        )

    def test_no_labeled_data_falls_back_to_prior_without_fitting(self):
        estimator = self._CallRecordingEstimator()
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            proba_format="array",
        )
        y_unlabeled = np.full_like(self.y_ml, -1)

        with self.assertWarnsRegex(UserWarning, "no labeled data"):
            clf.fit(self.X_ml, y_unlabeled)

        self.assertFalse(clf.is_fitted_)
        self.assertEqual(clf.estimator_.calls, [])
        np.testing.assert_allclose(clf.predict_proba(self.X_ml), 0.5)

    def test_degenerate_class_fit_failure_falls_back_to_prior(self):
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            missing_label="nan",
            classes=["tokyo", "paris"],
            random_state=0,
        )

        with self.assertWarnsRegex(UserWarning, "fewer than two"):
            clf.fit(self.fit_default_params["X"], self.y2)

        self.assertFalse(clf.is_fitted_)

    def test_unexpected_fit_failure_propagates_with_context(self):
        clf = SklearnClassifier(
            estimator=self._BrokenEstimator(),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
        )
        attributes_before = dict(clf.__dict__)

        with self.assertRaises(RuntimeError) as context:
            clf.fit(self.X_ml, self.y_ml)

        self.assertIn("_BrokenEstimator", str(context.exception))
        self.assertIn("fit", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, RuntimeError)
        self.assertEqual(
            "the estimator is broken", str(context.exception.__cause__)
        )
        # A suppressed failure must not leave a wrapper that passes the fitted
        # check and then silently serves prior-only predictions.
        assert_attributes_unchanged(self, clf, attributes_before)
        self.assertRaises(NotFittedError, check_is_fitted, clf)

    def test_failed_refit_preserves_previously_fitted_state(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
        ).fit(self.X_ml, self.y_ml)
        attributes_before = dict(clf.__dict__)
        expected_probabilities = clf.predict_proba(self.X_ml)

        clf.estimator = self._BrokenEstimator()
        with self.assertRaises(RuntimeError):
            clf.fit(self.X_ml, self.y_ml)

        assert_attributes_unchanged(
            self, clf, attributes_before, ignored={"estimator"}
        )
        np.testing.assert_allclose(
            clf.predict_proba(self.X_ml), expected_probabilities
        )

    def test_rejections_before_the_estimator_fit_are_transactional(self):
        # The snapshot taken by `_fit` has to cover every rejection raised
        # between the snapshot and the estimator call, not only a failing
        # estimator fit. The target specification is resolved before the first
        # attribute is written, so the first three cases were already safe and
        # are covered here against regression. The last four each used to
        # commit fitted attributes, the final three including `n_features_in_`.
        multilabel_params = {"classes": [[0, 1], [0, 1]], "missing_label": -1}
        cases = {
            "rejected class vocabulary": {
                "fit_params": {
                    "X": self.fit_default_params["X"],
                    "y": ["tokyo", "paris", "nan", "berlin"],
                },
                "error": ValueError,
                "message": "outside",
            },
            "rejected target specification": {
                "init_params": multilabel_params,
                "fit_params": {"X": self.X_ml, "y": self.y_ml},
                "error": ValueError,
                "message": "does not support multi-label classification",
            },
            "one-dimensional multi-label target": {
                "init_params": multilabel_params,
                "fit_params": {"X": self.X_ml, "y": self.y_ml[:, 0]},
                "error": ValueError,
                "message": "must be two-dimensional",
            },
            "inconsistent sample counts": {
                "fit_params": {
                    "X": self.fit_default_params["X"],
                    "y": self.fit_default_params["y"][:-1],
                },
                "error": ValueError,
                "message": "inconsistent numbers of samples",
            },
            "non-classifier estimator": {
                "init_params": {"estimator": LinearRegression()},
                "error": TypeError,
                "message": "must be a scikit-learn",
            },
            "non-boolean flag": {
                "init_params": {"include_unlabeled_samples": "yes"},
                "error": TypeError,
                "message": "include_unlabeled_samples",
            },
            "cost matrix without probabilities": {
                "init_params": {
                    "estimator": Perceptron(),
                    "cost_matrix": [[0, 1], [1, 0]],
                },
                "error": ValueError,
                "message": "'cost_matrix' can be only set",
            },
        }
        default_init_params = {
            "estimator": GaussianNB(),
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                clf = SklearnClassifier(
                    **{**default_init_params, **case.get("init_params", {})}
                )
                fit_params = case.get("fit_params", self.fit_default_params)

                assert_fit_failure_is_transactional(
                    self,
                    clf,
                    lambda: clf.fit(**fit_params),
                    case["error"],
                    case["message"],
                )
                self.assertRaises(NotFittedError, check_is_fitted, clf)

    def test_failed_validation_refit_preserves_previously_fitted_state(self):
        # A rejection raised after `_validate_data` used to leave the widened
        # `n_features_in_` behind. Because `is_fitted_` had been committed by
        # the earlier successful fit, the wrapper stayed fitted and could no
        # longer predict on the data it was trained on.
        clf = SklearnClassifier(
            estimator=GaussianNB(),
            classes=["tokyo", "paris"],
            missing_label="nan",
        ).fit(**self.fit_default_params)
        expected_predictions = clf.predict(**self.predict_default_params)

        clf.set_params(estimator=LinearRegression())
        assert_fit_failure_is_transactional(
            self,
            clf,
            lambda: clf.fit(np.zeros((4, 3)), self.fit_default_params["y"]),
            TypeError,
            "must be a scikit-learn",
        )

        self.assertTrue(clf.is_fitted_)
        self.assertEqual(clf.n_features_in_, 1)
        np.testing.assert_array_equal(
            clf.predict(**self.predict_default_params), expected_predictions
        )

    def test_failed_validation_partial_fit_preserves_fitted_state(self):
        # `partial_fit` reaches the same rejections through its own entry
        # point, and keeps `n_features_in_` because it does not reset the
        # feature count. The equal-valued `check_X_dict_` it writes is a new
        # object, so only the restored snapshot satisfies the identity check.
        clf = SklearnClassifier(
            estimator=GaussianNB(),
            classes=["tokyo", "paris"],
            missing_label="nan",
        ).partial_fit(**self.fit_default_params)
        expected_predictions = clf.predict(**self.predict_default_params)

        clf.set_params(include_unlabeled_samples="yes")
        assert_fit_failure_is_transactional(
            self,
            clf,
            lambda: clf.partial_fit(**self.fit_default_params),
            TypeError,
            "include_unlabeled_samples",
        )

        self.assertTrue(clf.is_fitted_)
        np.testing.assert_array_equal(
            clf.predict(**self.predict_default_params), expected_predictions
        )

    def test_object_encoded_targets_reach_the_estimator(self):
        # `missing_label=None` decodes into an `object` array that a
        # `scikit-learn` estimator rejects, which used to be swallowed by the
        # prior-only fallback instead of fitting the estimator.
        clf = SklearnClassifier(
            estimator=GaussianNB(), classes=[0, 1], missing_label=None
        )

        clf.fit(np.zeros((3, 1)), [0, None, 1])

        self.assertTrue(clf.is_fitted_)
        np.testing.assert_array_equal(clf.estimator_.classes_, [0, 1])

    def test_multilabel_predict_rejects_single_output_predictions(self):
        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((len(self.X_ml), 2), 0.5)
        )

        self.assertRaisesRegex(
            ValueError,
            r"`\(n_samples, 2\)`",
            clf.predict,
            self.X_ml,
        )

    def test_multilabel_predict_rejects_wrong_sample_count(self):
        clf = self._prefit_multilabel_clf()
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((len(self.X_ml), 2), 0.5),
            predictions=np.zeros((len(self.X_ml) - 1, 2), dtype=int),
        )

        self.assertRaisesRegex(
            ValueError,
            r"exactly `\(4, 2\)`",
            clf.predict,
            self.X_ml,
        )

    def test_multilabel_predict_rejects_labels_outside_output_vocabulary(self):
        clf = self._prefit_multilabel_clf()
        predictions = self.y_ml.copy()
        predictions[0, 1] = 2
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((len(self.X_ml), 2), 0.5),
            predictions=predictions,
        )

        self.assertRaisesRegex(
            ValueError,
            "Class 2.*output 1.*not contained",
            clf.predict,
            self.X_ml,
        )

    def test_multilabel_predict_proba_list_rejects_malformed_array(self):
        clf = self._prefit_multilabel_clf(proba_format="list")
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((len(self.X_ml), 3), 1 / 3)
        )

        self.assertRaisesRegex(
            ValueError,
            r"`\(n_samples, 2\)`",
            clf.predict_proba,
            self.X_ml,
        )

    def test_multilabel_predict_proba_accepts_nested_list_matrix(self):
        expected = np.array(
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ]
        )
        for proba_format in ["array", "list"]:
            with self.subTest(proba_format=proba_format):
                clf = SklearnClassifier(
                    estimator=self._NestedListProbaEstimator(),
                    classes=[[0, 1], [0, 1]],
                    missing_label=-1,
                    proba_format=proba_format,
                ).fit(self.X_ml, self.y_ml)

                probabilities = clf.predict_proba(self.X_ml)

                if proba_format == "array":
                    np.testing.assert_allclose(probabilities, expected)
                else:
                    self.assertEqual(len(probabilities), expected.shape[1])
                    for j, probabilities_j in enumerate(probabilities):
                        np.testing.assert_allclose(
                            probabilities_j,
                            np.column_stack(
                                [1 - expected[:, j], expected[:, j]]
                            ),
                        )

    def test_multilabel_predict_proba_array_rejects_wrong_sample_count(self):
        clf = self._prefit_multilabel_clf(proba_format="array")
        clf.estimator_ = self._PredictProbaEstimator(
            proba=np.full((len(self.X_ml) - 1, 2), 0.5)
        )

        self.assertRaisesRegex(
            ValueError,
            r"exactly `\(4, 2\)`",
            clf.predict_proba,
            self.X_ml,
        )

    def test_multilabel_predict_proba_array_rejects_invalid_values(self):
        invalid_values = [np.inf, -0.1, 1.1]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                probabilities = np.full((len(self.X_ml), 2), 0.5)
                probabilities[0, 0] = invalid_value
                clf = self._prefit_multilabel_clf(proba_format="array")
                clf.estimator_ = self._PredictProbaEstimator(
                    proba=probabilities
                )

                self.assertRaisesRegex(
                    ValueError,
                    "'probas' are invalid",
                    clf.predict_proba,
                    self.X_ml,
                )

    def test_multilabel_predict_proba_list_rejects_invalid_distributions(self):
        invalid_rows = [
            [np.inf, 0.0],
            [-0.1, 1.1],
            [0.4, 0.4],
            [np.nan, 1.1],
        ]
        for invalid_row in invalid_rows:
            with self.subTest(invalid_row=invalid_row):
                probabilities = [
                    np.full((len(self.X_ml), 2), 0.5),
                    np.full((len(self.X_ml), 2), 0.5),
                ]
                probabilities[0][0] = invalid_row
                clf = self._prefit_multilabel_clf(proba_format="list")
                clf.estimator_ = self._PredictProbaEstimator(
                    proba=probabilities
                )

                self.assertRaisesRegex(
                    ValueError,
                    "'probas' are invalid",
                    clf.predict_proba,
                    self.X_ml,
                )

    def test_explicit_multilabel_uses_resolved_order_for_public_outputs(self):
        X = np.arange(12, dtype=float).reshape(-1, 2)
        y = np.array(
            [
                [7.0, 4.0],
                [3.0, -2.0],
                [7.0, 4.0],
                [7.0, 4.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(DummyClassifier(strategy="prior")),
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        ).fit(X, y)

        probabilities = clf.predict_proba(X)
        predictions = clf.predict(X)

        self.assertEqual(clf.target_spec_.classes, ((3.0, 7.0), (-2.0, 4.0)))
        np.testing.assert_allclose(probabilities, 0.75)
        np.testing.assert_array_equal(
            predictions, np.tile([7.0, 4.0], (len(X), 1))
        )
        self.assertEqual(clf.score(X[:4], y[:4]), 0.75)

    def test_degenerate_multilabel_fallback_uses_declared_class_order(self):
        # The second output carries a single observed class, which
        # `GaussianProcessClassifier` rejects, so the wrapper degrades to the
        # class label distribution of the declared class vocabularies.
        X = np.arange(12, dtype=float).reshape(-1, 2)
        y = np.array(
            [
                [7.0, 4.0],
                [3.0, 4.0],
                [7.0, 4.0],
                [7.0, 4.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianProcessClassifier()),
            classes=[[3.0, 7.0], [-2.0, 4.0]],
            proba_format="array",
            random_state=0,
        )

        with self.assertWarnsRegex(UserWarning, "fewer than two"):
            clf.fit(X, y)
        probabilities = clf.predict_proba(X)
        predictions = clf.predict(X)

        self.assertFalse(clf.is_fitted_)
        np.testing.assert_allclose(probabilities[:, 0], 0.75)
        np.testing.assert_allclose(probabilities[:, 1], 1.0)
        self.assertTrue(np.isin(predictions[:, 0], [3.0, 7.0]).all())
        np.testing.assert_allclose(predictions[:, 1], 4.0)

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

        self.assertEqual(clf.target_spec_.target_type, "multi-label")
        for classes, expected_classes in zip(clf.classes_, estimator.classes_):
            np.testing.assert_array_equal(classes, expected_classes)
        self.assertEqual(P.shape, self.y_ml.shape)
        self.assertEqual(y_pred.shape, self.y_ml.shape)

        clf = SklearnClassifier(
            estimator=estimator, classes=None, proba_format="list"
        )
        P_list = clf.predict_proba(self.X_ml)

        self.assertEqual(clf.target_spec_.target_type, "multi-label")
        self.assertEqual(len(P_list), self.y_ml.shape[1])
        for P_j in P_list:
            self.assertEqual(P_j.shape, (len(self.X_ml), 2))

    def test_prefit_single_output_infers_classes(self):
        estimator = GaussianNB().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(estimator=estimator, classes=None)

        P = clf.predict_proba(self.X_ml)

        self.assertEqual(clf.target_spec_.target_type, "single-output")
        np.testing.assert_array_equal(clf.classes_, estimator.classes_)
        self.assertEqual(P.shape, (len(self.X_ml), len(estimator.classes_)))

    def _assert_prefit_rejection(self, clf, expected_message):
        attributes_before = dict(clf.__dict__)

        with self.assertRaisesRegex(ValueError, expected_message):
            check_is_fitted(clf)

        assert_attributes_unchanged(self, clf, attributes_before)

    def test_prefit_rejects_disjoint_equal_width_class_vocabulary(self):
        # Both vocabularies are two classes wide, so the contradiction cannot
        # be found by comparing their widths.
        estimator = LogisticRegression().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator, classes=[2, 3], missing_label=-1
        )

        self._assert_prefit_rejection(clf, "learned the class labels")

    def test_prefit_rejects_learned_classes_outside_configuration(self):
        estimator = LogisticRegression().fit(
            np.vstack([self.X_ml, [[3.0]]]), [0, 0, 1, 1, 2]
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1], missing_label=-1
        )

        self._assert_prefit_rejection(clf, r"learned the class labels \[2\]")

    def _multilabel_object_targets(self, output_values):
        """Build an object-valued multi-label target from per-output values."""
        y = np.empty(self.y_ml.shape, dtype=object)
        for output_idx, values in enumerate(output_values):
            y[:, output_idx] = [
                values[label] for label in self.y_ml[:, output_idx]
            ]
        return y

    def test_fit_rejects_heterogeneous_output_vocabularies(self):
        # A multi-label target is one array, so outputs declaring different
        # dtypes cannot be represented. The rejection precedes any comparison
        # against `y`, and above all any fitted state.
        y = self._multilabel_object_targets((("no", "yes"), (0, 1)))

        for classes in (
            [["no", "yes"], [0, 1]],
            [["no", "yes"], [0.0, 1.0]],
            [[0, 1], [0.0, 1.0]],
        ):
            with self.subTest(classes=classes):
                clf = SklearnClassifier(
                    MultiOutputClassifier(LogisticRegression()),
                    classes=classes,
                    target_type="multi-label",
                    missing_label=None,
                )

                assert_fit_failure_is_transactional(
                    self,
                    clf,
                    lambda: clf.fit(self.X_ml, y),
                    ValueError,
                    "one dtype across all label outputs",
                )

    def test_fit_accepts_homogeneous_non_numeric_vocabularies(self):
        # Homogeneous string vocabularies of differing width stay valid, and
        # an unordered declaration resolves to the canonical order.
        y = self._multilabel_object_targets((("no", "yes"), ("off", "always")))

        clf = SklearnClassifier(
            MultiOutputClassifier(LogisticRegression()),
            classes=[["yes", "no"], ["off", "always"]],
            target_type="multi-label",
            missing_label=None,
        ).fit(self.X_ml, y)

        self._assert_prefit_multilabel_consistency(
            clf, (("no", "yes"), ("always", "off"))
        )

    def _assert_prefit_multilabel_consistency(self, clf, expected_classes):
        # `predict`, `predict_proba`, `classes_`, and `target_spec_` have to
        # describe the same vocabularies, column order, and output count.
        P = clf.predict_proba(self.X_ml)
        y_pred = clf.predict(self.X_ml)

        self.assertEqual(clf.target_spec_.target_type, "multi-label")
        self.assertEqual(clf.target_spec_.classes, expected_classes)
        self.assertEqual(len(clf.classes_), len(expected_classes))
        for classes_j, expected_classes_j in zip(
            clf.classes_, expected_classes
        ):
            np.testing.assert_array_equal(classes_j, expected_classes_j)
        self.assertEqual(P.shape, (len(self.X_ml), len(expected_classes)))
        self.assertEqual(y_pred.shape, (len(self.X_ml), len(expected_classes)))
        for output_idx, expected_classes_j in enumerate(expected_classes):
            self.assertTrue(
                np.isin(y_pred[:, output_idx], expected_classes_j).all()
            )

    def test_prefit_multilabel_accepts_learned_output_vocabularies(self):
        estimators = {
            "MultiOutputClassifier": MultiOutputClassifier(GaussianNB()),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=5, random_state=0
            ),
        }
        for name, estimator in estimators.items():
            with self.subTest(estimator=name):
                clf = SklearnClassifier(
                    estimator=estimator.fit(self.X_ml, self.y_ml),
                    classes=[[0, 1], [0, 1]],
                    missing_label=-1,
                    proba_format="array",
                )

                self._assert_prefit_multilabel_consistency(
                    clf, ((0, 1), (0, 1))
                )

    def test_prefit_multilabel_accepts_declared_label_outputs(self):
        # `OneVsRestClassifier` publishes flat label-output identifiers plus
        # explicit multi-label metadata, so the declared vocabularies supply
        # each output's binary classes.
        estimator = OneVsRestClassifier(LogisticRegression()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            proba_format="array",
        )

        self._assert_prefit_multilabel_consistency(clf, ((0, 1), (0, 1)))

    def test_prefit_rejects_relabeled_indicator_outputs(self):
        # The estimator predicts a binary indicator per output, so its outputs
        # cannot be declared to carry other class labels. Fitting the same
        # configuration through the wrapper fails as well, because
        # `OneVsRestClassifier` rejects a two-dimensional string target.
        estimator = OneVsRestClassifier(LogisticRegression()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[["a", "b"], ["a", "b"]],
            missing_label="nan",
        )

        self._assert_prefit_rejection(clf, "binary indicator per label output")

    def test_prefit_flat_classes_identify_label_outputs(self):
        estimator = OneVsRestClassifier(LogisticRegression()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=None, missing_label=-1
        )

        # The flat learned classes `[0, 1]` identify two label outputs and
        # must not be read as one binary class vocabulary.
        self._assert_prefit_rejection(clf, "2 label outputs")

    def test_prefit_rejects_label_outputs_as_single_output(self):
        estimator = OneVsRestClassifier(LogisticRegression()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1], missing_label=-1
        )

        self._assert_prefit_rejection(
            clf, "cannot be declared as a single-output classifier"
        )

    def test_prefit_rejects_mismatched_label_output_count(self):
        y_ml = np.column_stack([self.y_ml, self.y_ml[:, 0]])
        estimators = {
            "MultiOutputClassifier": MultiOutputClassifier(GaussianNB()),
            "OneVsRestClassifier": OneVsRestClassifier(LogisticRegression()),
        }
        for name, estimator in estimators.items():
            with self.subTest(estimator=name):
                clf = SklearnClassifier(
                    estimator=estimator.fit(self.X_ml, y_ml),
                    classes=[[0, 1], [0, 1]],
                    missing_label=-1,
                )

                self._assert_prefit_rejection(clf, "3 label outputs")

    def test_prefit_rejects_learned_classes_outside_output_vocabulary(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=[[0, 1], [2, 3]], missing_label=-1
        )

        self._assert_prefit_rejection(
            clf, "learned the class labels .* for label output 1"
        )

    def test_prefit_rejects_single_output_estimator_as_multilabel(self):
        # `RandomForestClassifier` declares both multi-label admission tags,
        # so only its fitted target evidence reveals the contradiction.
        estimator = RandomForestClassifier(n_estimators=5, random_state=0).fit(
            self.X_ml, self.y_ml[:, 0]
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=[[0, 1], [0, 1]], missing_label=-1
        )

        self._assert_prefit_rejection(
            clf, "one categorical class assignment per sample"
        )

    def test_prefit_configured_superset_zero_fills_probabilities(self):
        estimator = LogisticRegression().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1, 2], missing_label=-1
        )

        P = clf.predict_proba(self.X_ml)
        y_pred = clf.predict(self.X_ml)

        self.assertEqual(clf.target_spec_.classes, (0, 1, 2))
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        self.assertEqual(P.shape, (len(self.X_ml), 3))
        np.testing.assert_allclose(P[:, 2], 0.0)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            P[:, :2], estimator.predict_proba(self.X_ml)
        )
        self.assertTrue(np.isin(y_pred, clf.classes_).all())

    def test_prefit_discovery_does_not_call_prediction_methods(self):
        estimator = self._CallRecordingEstimator().fit(self.X_ml, self.y_ml)
        estimator.calls.clear()
        clf = SklearnClassifier(
            estimator=estimator, classes=[[0, 1], [0, 1]], missing_label=-1
        )

        check_is_fitted(clf)

        self.assertEqual(estimator.calls, [])
        self.assertEqual(clf.target_spec_.target_type, "multi-label")

    def test_prefit_without_learned_classes_requires_declared_classes(self):
        estimator = self._NoClassesEstimator().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(estimator=estimator, missing_label=-1)

        self._assert_prefit_rejection(
            clf, "exposes no learned class vocabulary"
        )

    def test_prefit_without_learned_classes_uses_declared_classes(self):
        estimator = self._NoClassesEstimator().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1], missing_label=-1
        )

        P = clf.predict_proba(self.X_ml)

        self.assertEqual(clf.target_spec_.classes, (0, 1))
        np.testing.assert_allclose(P, np.tile([0.4, 0.6], (len(self.X_ml), 1)))

    def _prefit_binary_estimator(self):
        return LogisticRegression().fit(self.X_ml, self.y_ml[:, 0])

    def test_prefit_classes_are_read_before_any_fitted_call(self):
        estimator = self._prefit_binary_estimator()
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1], missing_label=-1
        )

        classes = clf.classes_
        target_spec = clf.target_spec_
        clf.predict_proba(self.X_ml)

        np.testing.assert_array_equal(classes, [0, 1])
        self.assertEqual(target_spec.classes, (0, 1))
        np.testing.assert_array_equal(clf.classes_, classes)
        self.assertEqual(clf.target_spec_, target_spec)

    def test_prefit_wider_classes_are_read_before_any_fitted_call(self):
        estimator = self._prefit_binary_estimator()
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1, 2], missing_label=-1
        )

        # The estimator's learned vocabulary is narrower than the declared
        # one, so delegating `classes_` would answer with the wrong width.
        np.testing.assert_array_equal(estimator.classes_, [0, 1])
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        self.assertEqual(clf.target_spec_.classes, (0, 1, 2))
        self.assertEqual(clf.predict_proba(self.X_ml).shape[1], 3)

    def test_prefit_rejected_classes_never_answer_as_a_valid_vocabulary(self):
        estimator = self._prefit_binary_estimator()
        clf = SklearnClassifier(
            estimator=estimator, classes=[2, 3], missing_label=-1
        )

        for item in ("classes_", "target_spec_"):
            with self.subTest(item=item):
                with self.assertRaisesRegex(
                    AttributeError, "learned the class labels"
                ):
                    getattr(clf, item)
                self.assertFalse(hasattr(clf, item))

    def test_prefit_rejected_classes_fall_back_to_the_declared_ones(self):
        # The pool target preflight reads the vocabulary through this idiom.
        # A rejected resolution has to leave it with the declared vocabulary,
        # which is what a subsequent `fit` would establish, rather than with
        # the estimator's learned one, which the wrapper does not accept.
        estimator = self._prefit_binary_estimator()
        clf = SklearnClassifier(
            estimator=estimator, classes=[2, 3], missing_label=-1
        )

        classes = getattr(clf, "classes_", clf.classes)

        np.testing.assert_array_equal(classes, [2, 3])
        clf.fit(self.X_ml, np.array([2, 3, 2, 3]))
        np.testing.assert_array_equal(clf.classes_, [2, 3])

    def test_prefit_estimator_target_spec_does_not_shadow_the_wrappers(self):
        estimator = self._TargetSpecEstimator().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1, 2], missing_label=-1
        )

        self.assertEqual(estimator.target_spec_.classes, (0, 1))
        self.assertEqual(clf.target_spec_.classes, (0, 1, 2))

    def test_prefit_delegates_estimator_owned_attributes(self):
        estimator = self._prefit_binary_estimator()
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1, 2], missing_label=-1
        )

        np.testing.assert_array_equal(clf.coef_, estimator.coef_)
        self.assertEqual(clf.n_features_in_, estimator.n_features_in_)

    def test_unfitted_wrapper_refuses_its_own_fitted_attributes(self):
        clf = SklearnClassifier(
            estimator=LogisticRegression(), classes=[0, 1], missing_label=-1
        )

        for item in SklearnClassifier._own_fitted_attributes:
            with self.subTest(item=item):
                self.assertFalse(hasattr(clf, item))
                with self.assertRaises(NotFittedError):
                    getattr(clf, item)

    def test_prefit_estimator_marker_does_not_skip_wrapper_contract(self):
        estimator = self._prefit_binary_estimator()
        estimator.is_fitted_ = True
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1, 2], missing_label=-1
        )

        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        self.assertTrue(clf.is_fitted_)
        self.assertIn("estimator_", clf.__dict__)

    def test_prefit_nested_wrapper_resolves_its_own_vocabulary(self):
        inner = SklearnClassifier(
            estimator=LogisticRegression(), classes=[0, 1], missing_label=-1
        ).fit(self.X_ml, self.y_ml[:, 0])
        outer = SklearnClassifier(
            estimator=inner, classes=[0, 1, 2], missing_label=-1
        )

        np.testing.assert_array_equal(inner.classes_, [0, 1])
        np.testing.assert_array_equal(outer.classes_, [0, 1, 2])
        self.assertEqual(inner.target_spec_.classes, (0, 1))
        self.assertEqual(outer.target_spec_.classes, (0, 1, 2))

    def test_unmappable_probability_columns_are_rejected(self):
        estimator = self._NoClassesEstimator().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator, classes=[0, 1, 2], missing_label=-1
        )

        self.assertRaisesRegex(
            ValueError,
            "does not expose `classes_`",
            clf.predict_proba,
            self.X_ml,
        )

    def _prefit_single_output_clf(self, proba, classes_):
        """Return a fitted wrapper whose estimator returns `proba`."""
        clf = SklearnClassifier(
            estimator=GaussianNB().fit(self.X_ml, self.y_ml[:, 0]),
            classes=[0, 1],
            missing_label=-1,
        )
        clf.predict_proba(self.X_ml)
        clf.n_features_in_ = self.X_ml.shape[1]
        clf.estimator_ = self._PredictProbaEstimator(
            proba=proba, classes_=classes_
        )
        return clf

    def test_malformed_single_output_probabilities_are_rejected(self):
        # The estimator contradicts the requested samples or its own learned
        # classes, so its columns cannot be mapped to the declared ones.
        n_samples = len(self.X_ml)
        malformed = {
            "too few samples": (np.full((1, 2), 0.5), r"`\(4, n_classes\)`"),
            "not two-dimensional": (
                np.full(n_samples, 0.5),
                r"`\(4, n_classes\)`",
            ),
            "column count": (
                np.full((n_samples, 3), 1 / 3),
                "reports 2 classes",
            ),
        }
        for reason, (proba, message) in malformed.items():
            with self.subTest(reason=reason):
                clf = self._prefit_single_output_clf(
                    proba=proba, classes_=np.array([0, 1])
                )

                self.assertRaisesRegex(
                    ValueError, message, clf.predict_proba, self.X_ml
                )

    def test_unlearned_probability_column_class_is_rejected(self):
        clf = self._prefit_single_output_clf(
            proba=np.full((len(self.X_ml), 1), 1.0), classes_=np.array([5])
        )

        self.assertRaisesRegex(
            ValueError,
            "Class 5 learned by the wrapped estimator is not contained",
            clf.predict_proba,
            self.X_ml,
        )

    def test_prefit_maps_probability_columns_by_class_identity(self):
        estimator = self._ReorderedClassesEstimator().fit(
            self.X_ml, self.y_ml[:, 0]
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=None, missing_label=-1
        )

        P = clf.predict_proba(self.X_ml)

        # The learned order is normalized to the documented class order, so
        # the estimator's leading column of class 1 becomes the last column.
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        self.assertEqual(clf.target_spec_.classes, (0, 1))
        np.testing.assert_allclose(P, np.tile([0.2, 0.8], (len(self.X_ml), 1)))

    def test_prefit_cost_sensitive_prediction_uses_mapped_columns(self):
        estimator = self._ReorderedClassesEstimator().fit(
            self.X_ml, self.y_ml[:, 0]
        )
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[0, 1],
            cost_matrix=1 - np.eye(2),
            missing_label=-1,
        )

        y_pred = clf.predict(self.X_ml)

        # Class `1` carries the estimator's probability of `0.8`, so it has the
        # lowest expected costs once the columns are mapped by class identity.
        np.testing.assert_array_equal(y_pred, np.ones(len(self.X_ml)))

    def test_prefit_initialization_commits_complete_label_state(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator, classes=None, missing_label=-1
        )

        check_is_fitted(clf)

        self.assertTrue(clf.__dict__["is_fitted_"])
        self.assertIn("estimator_", clf.__dict__)
        self.assertIn("check_X_dict_", clf.__dict__)
        self.assertEqual(clf.target_spec_.target_type, "multi-label")
        self.assertEqual(clf.target_spec_.classes, ((0, 1), (0, 1)))
        self.assertIsNone(clf.cost_matrix_)
        for classes_j, expected_classes_j in zip(
            clf.classes_, estimator.classes_
        ):
            np.testing.assert_array_equal(classes_j, expected_classes_j)
        for counts_j, classes_j in zip(clf._label_counts, clf.classes_):
            np.testing.assert_array_equal(counts_j, np.zeros(len(classes_j)))

    def test_prefit_multilabel_rejects_cost_matrix_without_state_changes(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            cost_matrix=np.eye(3),
        )
        attributes_before = dict(clf.__dict__)

        with self.assertRaisesRegex(ValueError, "cost_matrix"):
            clf.predict_proba(self.X_ml)

        assert_attributes_unchanged(self, clf, attributes_before)

    def test_prefit_target_resolution_failure_commits_no_fitted_state(self):
        estimator = GaussianNB().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator, missing_label=-1, target_type="multi-label"
        )
        attributes_before = dict(clf.__dict__)

        with self.assertRaisesRegex(
            ValueError, "nested binary class vocabularies"
        ):
            check_is_fitted(clf)

        assert_attributes_unchanged(self, clf, attributes_before)

    def test_prefit_capability_failure_commits_no_fitted_state(self):
        estimator = Perceptron().fit(self.X_ml, self.y_ml[:, 0])
        clf = SklearnClassifier(
            estimator=estimator,
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        )
        attributes_before = dict(clf.__dict__)

        with self.assertRaisesRegex(ValueError, "does not support"):
            check_is_fitted(clf)

        assert_attributes_unchanged(self, clf, attributes_before)

    def test_failed_recheck_preserves_initialized_label_state(self):
        clf = SklearnClassifier(estimator=Perceptron(), missing_label=-1)
        clf._commit_label_state(clf._resolve_label_state(np.array([0, 1])))
        attributes_before = dict(clf.__dict__)

        with self.assertRaisesRegex(ValueError, "does not support"):
            clf._commit_label_state(clf._resolve_label_state([[0, 1], [0, 1]]))

        assert_attributes_unchanged(self, clf, attributes_before)
        self.assertEqual(clf.target_spec_.target_type, "single-output")
        np.testing.assert_array_equal(clf.classes_, [0, 1])

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

    def test_prefit_multilabel_nan_falls_back_to_uniform_prior(self):
        estimator = MultiOutputClassifier(GaussianNB()).fit(
            self.X_ml, self.y_ml
        )

        def predict_proba_nan(X, **kwargs):
            return [
                np.full((len(X), len(classes_j)), np.nan)
                for classes_j in estimator.classes_
            ]

        estimator.predict_proba = predict_proba_nan
        for proba_format in ["array", "list"]:
            with self.subTest(proba_format=proba_format):
                clf = SklearnClassifier(
                    estimator=estimator,
                    classes=None,
                    missing_label=-1,
                    proba_format=proba_format,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    P = clf.predict_proba(self.X_ml)

                if proba_format == "array":
                    self.assertEqual(P.shape, self.y_ml.shape)
                else:
                    self.assertEqual(len(P), self.y_ml.shape[1])
                    for P_j in P:
                        self.assertEqual(P_j.shape, (len(self.X_ml), 2))
                self.assertFalse(np.any(np.isnan(P)))
                np.testing.assert_allclose(P, np.full_like(P, 0.5))

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
        clf = self._prefit_multilabel_clf()
        clf.proba_format = "invalid"
        self.assertRaises(ValueError, clf._resolve_proba_format)

        clf.proba_format = "auto"
        self.assertEqual(clf._resolve_proba_format(), "array")

        clf.proba_format = "array"
        self.assertEqual(clf._resolve_proba_format(), "array")

        clf.proba_format = "list"
        self.assertEqual(clf._resolve_proba_format(), "list")

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


class _IncrementalClassifierTargetContract:
    def _make_incremental_contract_classifier(self):
        raise NotImplementedError

    def _capture_incremental_contract_details(self, clf):
        return None

    def _assert_incremental_contract_details(
        self, clf, established_spec, initial_details
    ):
        pass

    def test_incremental_target_contract_preserves_vocabulary_on_subset(self):
        clf = self._make_incremental_contract_classifier()
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        initial_details = self._capture_incremental_contract_details(clf)

        clf.partial_fit(X, np.array([0, 0]))

        self.assertIs(clf.target_spec_, established_spec)
        self.assertEqual(clf.target_spec_.classes, (0, 1))
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        self.assertEqual(clf.predict_proba(X).shape, (2, 2))
        self._assert_incremental_contract_details(
            clf, established_spec, initial_details
        )

    def test_incremental_target_contract_rejects_unseen_class_before_training(
        self,
    ):
        clf = self._make_incremental_contract_classifier()
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        probabilities = clf.predict_proba(X)

        with self.assertRaisesRegex(ValueError, "class"):
            clf.partial_fit(X[:1], np.array([2]))

        self.assertIs(clf.target_spec_, established_spec)
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        np.testing.assert_array_equal(clf.predict_proba(X), probabilities)

    def test_incremental_target_contract_accepts_unlabeled_subset(self):
        clf = self._make_incremental_contract_classifier()
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.partial_fit(X, np.array([-1, -1]))

        self.assertIs(clf.target_spec_, established_spec)
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probabilities = clf.predict_proba(X)
        self.assertEqual(probabilities.shape, (2, 2))

    def test_incremental_contract_rejects_target_type_change_before_training(
        self,
    ):
        clf = self._make_incremental_contract_classifier()
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        probabilities = clf.predict_proba(X)

        clf.target_type = "multi-label"
        with self.assertRaises(ValueError):
            clf.partial_fit(X, np.array([[0, 1], [1, 0]]))
        clf.target_type = "auto"

        self.assertIs(clf.target_spec_, established_spec)
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        np.testing.assert_array_equal(clf.predict_proba(X), probabilities)


class _StreamingClassifierEmptyUpdateContract:
    def test_incremental_empty_training_subsets_preserve_state(self):
        clf = self._make_incremental_contract_classifier()
        X = np.array([[0.0], [1.0]])
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        established_estimator = clf.estimator_
        established_counts = deepcopy(clf._label_counts)
        established_probabilities = clf.predict_proba(X)

        updates = [
            (np.empty((0, 1)), np.empty(0, dtype=int)),
            (X, np.array([-1, -1])),
        ]
        for X_update, y_update in updates:
            with self.subTest(update_size=len(y_update)):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    clf.partial_fit(X_update, y_update)

                self.assertTrue(clf.is_fitted_)
                self.assertIs(clf.target_spec_, established_spec)
                self.assertIs(clf.estimator_, established_estimator)
                np.testing.assert_array_equal(
                    clf._label_counts, established_counts
                )
                np.testing.assert_allclose(
                    clf.predict_proba(X), established_probabilities
                )


class TestSlidingWindowClassifier(
    _IncrementalClassifierTargetContract,
    TemplateSkactivemlClassifier,
    unittest.TestCase,
):
    non_integral_classes_error = RuntimeError

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

    def _make_incremental_contract_classifier(self):
        return SlidingWindowClassifier(
            estimator=SklearnClassifier(GaussianNB(), missing_label=-1),
            missing_label=-1,
            window_size=2,
        )

    def _capture_incremental_contract_details(self, clf):
        return clf.estimator_

    def _assert_incremental_contract_details(
        self, clf, established_spec, initial_estimator
    ):
        self.assertIsNot(clf.estimator_, initial_estimator)
        self.assertEqual(clf.estimator_.target_spec_, established_spec)
        np.testing.assert_array_equal(clf.estimator_.classes_, [0, 1])

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

    def test_predict_dtype_matches_class_dtype(
        self, replace_init_params=None, replace_fit_params=None
    ):
        # The wrapped classifier must agree on `missing_label`, so the
        # default estimator pinned to `"nan"` is replaced.
        init_params = {
            "estimator": SklearnClassifier(
                GaussianProcessClassifier(), missing_label=np.nan
            )
        }
        if replace_init_params is not None:
            init_params.update(replace_init_params)
        super().test_predict_dtype_matches_class_dtype(
            replace_init_params=init_params,
            replace_fit_params=replace_fit_params,
        )

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

        # The wrapped `SklearnClassifier` forwards the non-integral float
        # class labels, which `LogisticRegression` rejects as continuous.
        test_cases = [
            ("state", TypeError),
            (0.0, self.non_integral_classes_error),
        ]
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

    def test_fit_rejects_mismatched_estimator_classes(self):
        estimator = SklearnClassifier(
            GaussianNB(), classes=[0, 1], missing_label=-1
        )
        clf = SlidingWindowClassifier(
            estimator=estimator, classes=[0, 2], missing_label=-1
        )

        with self.assertRaises(ValueError):
            clf.fit([[0.0], [1.0]], [0, 0])

    def test_unfitted_wrapper_refuses_its_own_fitted_attributes(self):
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                GaussianNB(), classes=[0, 1], missing_label=-1
            ),
            missing_label=-1,
        )

        for item in SlidingWindowClassifier._own_fitted_attributes:
            with self.subTest(item=item):
                self.assertFalse(hasattr(clf, item))
                with self.assertRaises(NotFittedError):
                    getattr(clf, item)

    def test_classes_stay_delegated_to_the_wrapped_classifier(self):
        # This wrapper resolves no class vocabulary of its own, so `classes_`
        # is not among its own fitted attributes and keeps being answered by
        # the `SkactivemlClassifier` it wraps.
        estimator = SklearnClassifier(
            GaussianNB(), classes=[0, 1], missing_label=-1
        ).fit([[0.0], [1.0]], [0, 1])
        clf = SlidingWindowClassifier(estimator=estimator, missing_label=-1)
        clf.fit([[0.0], [1.0]], [0, 1])

        self.assertNotIn("classes_", clf.__dict__)
        np.testing.assert_array_equal(clf.classes_, clf.estimator_.classes_)

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
            ([0, 1, 2, -1], ValueError),
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

    def _fitted_window_classifier(self):
        # `Perceptron` has no `predict_proba`, which the wrapped
        # `SklearnClassifier` forwards, so a `cost_matrix` set afterwards is
        # rejected inside `_fit`, after `_add_samples` extended the window.
        clf = SlidingWindowClassifier(
            estimator=SklearnClassifier(
                Perceptron(), classes=[0, 1], missing_label=-1
            ),
            classes=[0, 1],
            missing_label=-1,
            window_size=10,
        )
        return clf.fit(np.arange(8.0).reshape(4, 2), np.array([0, 1, 0, 1]))

    def test_failing_partial_fit_rolls_back_the_sliding_window(self):
        # `partial_fit` extends the window before it fits, so a rejection
        # inside `_fit` used to leave the window carrying samples the
        # estimator was never trained on. The rejected update carries labels
        # the previous window does not, so a window that kept them fails here.
        X = np.arange(8.0).reshape(4, 2)
        clf = self._fitted_window_classifier()
        X_window_before = [np.copy(x) for x in clf.X_train_]
        y_window_before = list(clf.y_train_)
        clf.cost_matrix = [[0, 1], [1, 0]]

        assert_fit_failure_is_transactional(
            self,
            clf,
            lambda: clf.partial_fit(X, np.array([1, 1, 1, 1])),
            ValueError,
            "cost_matrix",
        )

        np.testing.assert_array_equal(list(clf.X_train_), X_window_before)
        np.testing.assert_array_equal(list(clf.y_train_), y_window_before)

    def test_failing_fit_rolls_back_the_sliding_window(self):
        # `fit` replaces the window rather than extending it, so the rollback
        # has to put the previous window back as well.
        X = np.arange(8.0).reshape(4, 2)
        clf = self._fitted_window_classifier()
        window_before = [np.copy(x) for x in clf.X_train_]
        clf.cost_matrix = [[0, 1], [1, 0]]

        assert_fit_failure_is_transactional(
            self,
            clf,
            lambda: clf.fit(X[:2], np.array([0, 1])),
            ValueError,
            "cost_matrix",
        )

        np.testing.assert_array_equal(list(clf.X_train_), window_before)

    def test_failing_estimator_fit_rolls_back_the_sliding_window(self):
        # The wrapped estimator's own failure is rolled back the same way.
        class FailingClassifier(ParzenWindowClassifier):
            def fit(self, X, y, sample_weight=None):
                raise RuntimeError("the wrapped classifier is broken")

        X = np.arange(8.0).reshape(4, 2)
        y = np.array([0, 1, 0, 1])
        clf = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(classes=[0, 1], missing_label=-1),
            classes=[0, 1],
            missing_label=-1,
            window_size=10,
        ).fit(X, y)
        window_before = [np.copy(x) for x in clf.X_train_]
        clf.estimator = FailingClassifier(classes=[0, 1], missing_label=-1)

        assert_fit_failure_is_transactional(
            self,
            clf,
            lambda: clf.partial_fit(X, y),
            RuntimeError,
            "the wrapped classifier is broken",
        )

        np.testing.assert_array_equal(list(clf.X_train_), window_before)

    def test_window_rollback_keeps_the_window_objects_themselves(self):
        # The rollback refills the deques rather than replacing them, so a
        # caller holding a reference to `X_train_` sees the rollback too.
        X = np.arange(8.0).reshape(4, 2)
        y = np.array([0, 1, 0, 1])
        clf = self._fitted_window_classifier()
        window = clf.X_train_
        clf.cost_matrix = [[0, 1], [1, 0]]

        with self.assertRaises(ValueError):
            clf.partial_fit(X, y)

        self.assertIs(clf.X_train_, window)
        self.assertEqual(len(window), 4)
        self.assertEqual(window.maxlen, 10)

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

        def test_init_param_validate_proba(self):
            # The flag is consumed by `predict_proba`, so it is checked
            # there rather than while fitting.
            for validate_proba, err in [
                (True, None),
                (False, None),
                ("Test", TypeError),
                (None, TypeError),
            ]:
                with self.subTest(validate_proba=validate_proba):
                    init_params = self.init_default_params.copy()
                    init_params["validate_proba"] = validate_proba
                    clf = SkorchClassifier(**init_params).fit(self.X, self.y)
                    if err is None:
                        clf.predict_proba(self.X)
                    else:
                        self.assertRaises(err, clf.predict_proba, self.X)

        def test_predict_proba_rejects_values_that_are_no_probabilities(self):
            # `forward_outputs` decides how the module's outputs are read, so
            # a mapping without a suitable transform passes on raw scores.
            init_params = self.init_default_params.copy()
            init_params["forward_outputs"] = {"proba": (0, None)}
            clf = SkorchClassifier(**init_params).fit(self.X, self.y)

            with self.assertRaisesRegex(ValueError, "'probas' are invalid"):
                clf.predict_proba(self.X)

            init_params["validate_proba"] = False
            clf = SkorchClassifier(**init_params).fit(self.X, self.y)

            P = clf.predict_proba(self.X)

            self.assertFalse(
                np.allclose(np.sum(P, axis=1), 1, rtol=0, atol=1.0e-3)
            )

        def test_predict_proba_rejects_multilabel_output_count_mismatch(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "module": nn.Linear(1, 3),
                    "classes": [[0, 1], [0, 1]],
                    "missing_label": -1,
                }
            )
            clf = SkorchClassifier(**init_params)

            with self.assertRaisesRegex(
                ValueError, r"shape `\(n_samples, 2\)`.+got \(4, 3\)"
            ):
                clf.predict_proba(self.X_ml)

            self.assertFalse(hasattr(clf, "target_spec_"))
            self.assertFalse(hasattr(clf, "_le"))

        def test_prefit_multilabel_rejects_cost_matrix(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "module": nn.Linear(1, 2),
                    "classes": [[0, 1], [0, 1]],
                    "missing_label": -1,
                    "cost_matrix": np.eye(3),
                }
            )
            clf = SkorchClassifier(**init_params)

            with self.assertRaisesRegex(ValueError, "cost_matrix"):
                clf.predict_proba(self.X_ml)

            self.assertFalse(hasattr(clf, "target_spec_"))
            self.assertFalse(hasattr(clf, "_le"))
            self.assertFalse(hasattr(clf, "classes_"))
            self.assertFalse(hasattr(clf, "cost_matrix_"))

        def test_init_param_target_type(self):
            self._test_param(
                "init",
                "target_type",
                [
                    ("auto", None),
                    ("single-output", None),
                    ("multi-label", ValueError),
                    ("multi-output", ValueError),
                    ("invalid", ValueError),
                ],
            )

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

        def test_prefit_multilabel_prediction_supports_output_counts(self):
            for n_outputs in [1, 3]:
                with self.subTest(n_outputs=n_outputs):
                    init_params = deepcopy(self.init_default_params)
                    init_params.update(
                        {
                            "module": nn.Linear(1, n_outputs),
                            "classes": [[0, 1] for _ in range(n_outputs)],
                            "missing_label": -1,
                        }
                    )
                    clf = SkorchClassifier(**init_params)

                    probabilities = clf.predict_proba(self.X_ml)
                    predictions = clf.predict(self.X_ml)

                    expected_shape = (len(self.X_ml), n_outputs)
                    self.assertEqual(probabilities.shape, expected_shape)
                    self.assertEqual(predictions.shape, expected_shape)

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

        def test_multilabel_fit_includes_complete_unlabeled_rows(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "classes": [[0, 1], [0, 1]],
                    "missing_label": -1,
                    "target_type": "multi-label",
                    "include_unlabeled_samples": True,
                }
            )
            init_params["neural_net_param_dict"]["max_epochs"] = 1
            y = self.y_ml.copy()
            y[1] = -1
            clf = SkorchClassifier(**init_params)

            clf.fit(self.X_ml, y)

            check_is_fitted(clf)
            self.assertEqual(clf.target_spec_.target_type, "multi-label")
            self.assertEqual(clf.predict_proba(self.X_ml).shape, y.shape)

        def test_partial_fit_reuses_resolved_multilabel_target_spec(self):
            init_params = self.init_default_params.copy()
            init_params.update(
                {
                    "classes": None,
                    "missing_label": -1,
                    "target_type": "multi-label",
                }
            )
            clf = SkorchClassifier(**init_params)

            clf.partial_fit(self.X_ml, self.y_ml)
            established_spec = clf.target_spec_
            established_net = clf.neural_net_

            clf.partial_fit(
                self.X_ml[:2], np.array([[1, 0], [1, 0]], dtype=np.float32)
            )

            self.assertIs(clf.target_spec_, established_spec)
            self.assertIs(clf.neural_net_, established_net)
            self.assertEqual(clf.target_spec_.classes, ((0.0, 1.0),) * 2)
            for classes in clf.classes_:
                np.testing.assert_array_equal(classes, [0.0, 1.0])
            probabilities = clf.predict_proba(self.X_ml)
            self.assertTrue(np.all(probabilities >= 0))
            self.assertTrue(np.all(probabilities <= 1))

        def test_empty_partial_fit_reuses_resolved_multilabel_target_spec(
            self,
        ):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "classes": None,
                    "missing_label": -1,
                    "target_type": "multi-label",
                }
            )
            init_params["neural_net_param_dict"]["max_epochs"] = 1
            clf = SkorchClassifier(**init_params)
            clf.partial_fit(self.X_ml, self.y_ml)
            established_spec = clf.target_spec_
            established_net = clf.neural_net_
            established_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            established_history_length = len(clf.neural_net_.history)

            clf.partial_fit(
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

            self.assertIs(clf.target_spec_, established_spec)
            self.assertIs(clf.neural_net_, established_net)
            self.assertEqual(
                len(clf.neural_net_.history), established_history_length
            )
            np.testing.assert_array_equal(
                to_numpy(clf.neural_net_.module_.input_to_hidden.weight),
                established_weights,
            )

        def test_partial_fit_rejects_unseen_multilabel_class_before_training(
            self,
        ):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "classes": None,
                    "missing_label": -1,
                    "target_type": "multi-label",
                }
            )
            init_params["neural_net_param_dict"]["max_epochs"] = 1
            clf = SkorchClassifier(**init_params)
            clf.partial_fit(self.X_ml, self.y_ml)
            established_spec = clf.target_spec_
            established_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )

            invalid_y = self.y_ml[:1].copy()
            invalid_y[0, 0] = 2
            with self.assertRaisesRegex(ValueError, "class"):
                clf.partial_fit(self.X_ml[:1], invalid_y)

            self.assertIs(clf.target_spec_, established_spec)
            np.testing.assert_array_equal(
                to_numpy(clf.neural_net_.module_.input_to_hidden.weight),
                established_weights,
            )

        def test_warm_start_fit_reuses_target_spec(self):
            init_params = deepcopy(self.init_default_params)
            init_params["neural_net_param_dict"].update(
                {"max_epochs": 1, "warm_start": True}
            )
            clf = SkorchClassifier(**init_params)
            clf.fit(self.X, self.y_true)
            established_spec = clf.target_spec_
            established_net = clf.neural_net_

            clf.fit(self.X[:2], np.array([0, 0], dtype=np.float32))

            self.assertIs(clf.target_spec_, established_spec)
            self.assertIs(clf.neural_net_, established_net)
            np.testing.assert_array_equal(clf.classes_, [0, 1])

        def test_multilabel_warm_start_fit_reuses_target_spec(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "classes": None,
                    "missing_label": -1,
                    "target_type": "multi-label",
                }
            )
            init_params["neural_net_param_dict"].update(
                {"max_epochs": 1, "warm_start": True}
            )
            clf = SkorchClassifier(**init_params)
            clf.fit(self.X_ml, self.y_ml)
            established_spec = clf.target_spec_
            established_net = clf.neural_net_

            clf.fit(
                self.X_ml[:2],
                np.array([[0, 1], [0, 1]], dtype=np.float32),
            )

            self.assertIs(clf.target_spec_, established_spec)
            self.assertIs(clf.neural_net_, established_net)
            self.assertEqual(clf.target_spec_.classes, ((0.0, 1.0),) * 2)
            for classes in clf.classes_:
                np.testing.assert_array_equal(classes, [0.0, 1.0])

        def test_empty_multilabel_warm_start_fit_reuses_target_spec(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "classes": None,
                    "missing_label": -1,
                    "target_type": "multi-label",
                }
            )
            init_params["neural_net_param_dict"].update(
                {"max_epochs": 1, "warm_start": True}
            )
            clf = SkorchClassifier(**init_params)
            clf.fit(self.X_ml, self.y_ml)
            established_spec = clf.target_spec_
            established_net = clf.neural_net_
            established_weights = to_numpy(
                deepcopy(clf.neural_net_.module_.input_to_hidden.weight)
            )
            established_history_length = len(clf.neural_net_.history)

            clf.fit(
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

            self.assertIs(clf.target_spec_, established_spec)
            self.assertIs(clf.neural_net_, established_net)
            self.assertEqual(
                len(clf.neural_net_.history), established_history_length
            )
            np.testing.assert_array_equal(
                to_numpy(clf.neural_net_.module_.input_to_hidden.weight),
                established_weights,
            )

        def test_reinitializing_fit_resolves_new_target_spec(self):
            init_params = deepcopy(self.init_default_params)
            init_params["neural_net_param_dict"].update(
                {"max_epochs": 1, "warm_start": False}
            )
            clf = SkorchClassifier(**init_params)
            clf.fit(self.X, self.y_true)
            established_spec = clf.target_spec_
            established_net = clf.neural_net_

            clf.fit(self.X, self.y_true + 2)

            self.assertIsNot(clf.target_spec_, established_spec)
            self.assertIsNot(clf.neural_net_, established_net)
            self.assertEqual(clf.target_spec_.classes, (2, 3))
            np.testing.assert_array_equal(clf.classes_, [2, 3])

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

        def test_fit_infers_multilabel_target_from_public_input(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(
                {
                    "classes": None,
                    "missing_label": -1,
                    "target_type": "multi-label",
                }
            )
            init_params["neural_net_param_dict"]["max_epochs"] = 1
            clf = SkorchClassifier(**init_params)

            clf.fit(self.X_ml, self.y_ml)

            self.assertEqual(clf.target_spec_.target_type, "multi-label")
            self.assertEqual(
                clf.predict_proba(self.X_ml).shape, self.y_ml.shape
            )

        def test_predict_proba_initializes_public_fallback_cost_matrix(self):
            init_params = deepcopy(self.init_default_params)
            init_params["classes"] = [0, 1]
            clf = SkorchClassifier(**init_params)

            clf.predict_proba(self.X)

            np.testing.assert_array_equal(
                clf.cost_matrix_, 1 - np.eye(len(clf.classes_))
            )

        def test_multilabel_public_fallback_preserves_predictions(self):
            init_params = deepcopy(self.init_default_params)
            init_params.update(self.init_default_params_multilabel)
            init_params["neural_net_param_dict"]["max_epochs"] = 1
            clf = SkorchClassifier(**init_params)

            probabilities_before = clf.predict_proba(self.X_ml)
            clf.partial_fit(
                self.X_ml,
                np.full_like(self.y_ml, fill_value=-1),
            )
            probabilities_after = clf.predict_proba(self.X_ml)

            np.testing.assert_array_equal(
                probabilities_after, probabilities_before
            )

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
            self.assertEqual(
                clf_no_classes.target_spec_.target_type, "single-output"
            )

            clf_multilabel = SkorchClassifier(**init_params)
            clf_multilabel._initialize_fallbacks(np.zeros((2, 2)))
            self.assertEqual(
                clf_multilabel.target_spec_.target_type, "multi-label"
            )
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
            self._test_extra_outputs("predict")

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

    class TestRiverClassifier(
        _StreamingClassifierEmptyUpdateContract,
        _IncrementalClassifierTargetContract,
        TemplateSkactivemlClassifier,
        unittest.TestCase,
    ):
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

        def _make_incremental_contract_classifier(self):
            return RiverClassifier(
                estimator=river.naive_bayes.GaussianNB(),
                missing_label=-1,
                random_state=0,
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
                        # Incremental empty batches reuse the established
                        # classes; reinitializing fits still require an
                        # explicit class declaration.
                        if provide_classes or fit_function == "partial_fit":
                            fit_func(X[:0], y_true[:0])
                            self.assertEqual(
                                clf.is_fitted_, fit_function == "partial_fit"
                            )
                        else:
                            self.assertRaises(
                                ValueError, fit_func, X[:0], y_true[:0]
                            )
                        if provide_classes:
                            fit_func(X, y_all_missing)
                            self.assertEqual(
                                clf.is_fitted_, fit_function == "partial_fit"
                            )
                        elif fit_function == "fit":
                            self.assertRaises(
                                ValueError, fit_func, X, y_all_missing
                            )

        def test_fit(self):
            self._test_fit("fit")

        def test_wrong_estimator_rejection_is_transactional(self):
            # This wrapper absorbs every estimator failure into its prior-only
            # fallback, so the wrong-estimator rejection is nearly its only
            # raising path. It used to leak all seven fitted attributes,
            # `n_features_in_` among them.
            X = np.zeros((4, 2))
            y = np.array([0, 1, 0, 1])
            clf = RiverClassifier(
                estimator=LinearRegression(), classes=[0, 1], missing_label=-1
            )

            assert_fit_failure_is_transactional(
                self,
                clf,
                lambda: clf.fit(X, y),
                TypeError,
                "must be a river classifier",
            )
            self.assertRaises(NotFittedError, check_is_fitted, clf)

        def test_wrong_estimator_refit_preserves_fitted_state(self):
            X = np.arange(8.0).reshape(4, 2)
            y = np.array([0, 1, 0, 1])
            clf = RiverClassifier(
                estimator=river.naive_bayes.GaussianNB(),
                classes=[0, 1],
                missing_label=-1,
            ).fit(X, y)
            expected_probabilities = clf.predict_proba(X)

            clf.estimator = LinearRegression()
            assert_fit_failure_is_transactional(
                self,
                clf,
                lambda: clf.fit(np.zeros((4, 3)), y),
                TypeError,
                "must be a river classifier",
            )

            self.assertEqual(clf.n_features_in_, 2)
            np.testing.assert_allclose(
                clf.predict_proba(X), expected_probabilities
            )

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
        _StreamingClassifierEmptyUpdateContract,
        _IncrementalClassifierTargetContract,
        TemplateSkactivemlClassifier,
        unittest.TestCase,
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

        def _make_incremental_contract_classifier(self):
            from capymoa.classifier import AdaptiveRandomForestClassifier

            return CapyMOAClassifier(
                estimator_class=AdaptiveRandomForestClassifier,
                missing_label=-1,
                random_state=0,
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
                        # Incremental empty batches reuse the established
                        # classes; reinitializing fits still require an
                        # explicit class declaration.
                        if provide_classes or fit_function == "partial_fit":
                            fit_func(X[:0], y_true[:0])
                            self.assertEqual(
                                clf.is_fitted_, fit_function == "partial_fit"
                            )
                        else:
                            self.assertRaises(
                                ValueError, fit_func, X[:0], y_true[:0]
                            )
                        if provide_classes:
                            fit_func(X, y_all_missing)
                            self.assertEqual(
                                clf.is_fitted_, fit_function == "partial_fit"
                            )
                        elif fit_function == "fit":
                            self.assertRaises(
                                ValueError, fit_func, X, y_all_missing
                            )

        def test_fit(self):
            self._test_fit("fit")

        def test_wrong_estimator_class_rejection_is_transactional(self):
            # As for `RiverClassifier`, this rejection is nearly the only
            # raising path and used to leak all seven fitted attributes.
            X = np.zeros((4, 2))
            y = np.array([0, 1, 0, 1])
            clf = CapyMOAClassifier(
                estimator_class=LinearRegression,
                classes=[0, 1],
                missing_label=-1,
            )

            assert_fit_failure_is_transactional(
                self,
                clf,
                lambda: clf.fit(X, y),
                TypeError,
                "must be a capymoa",
            )
            self.assertRaises(NotFittedError, check_is_fitted, clf)

        def test_wrong_estimator_class_refit_preserves_fitted_state(self):
            from capymoa.classifier import NaiveBayes

            X = np.arange(8.0).reshape(4, 2)
            y = np.array([0, 1, 0, 1])
            clf = CapyMOAClassifier(
                estimator_class=NaiveBayes, classes=[0, 1], missing_label=-1
            ).fit(X, y)
            expected_probabilities = clf.predict_proba(X)

            clf.estimator_class = LinearRegression
            assert_fit_failure_is_transactional(
                self,
                clf,
                lambda: clf.fit(np.zeros((4, 3)), y),
                TypeError,
                "must be a capymoa",
            )

            self.assertEqual(clf.n_features_in_, 2)
            np.testing.assert_allclose(
                clf.predict_proba(X), expected_probabilities
            )

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
