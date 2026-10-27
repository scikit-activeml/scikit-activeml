import unittest
from itertools import product
from unittest.mock import PropertyMock, patch

import numpy as np
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import (
    ParzenWindowClassifier,
    SklearnClassifier,
    SlidingWindowClassifier,
)
from skactiveml.classifier.multiannotator import AnnotatorEnsembleClassifier
from skactiveml.pool import ContrastiveAL, UncertaintySampling
from skactiveml.pool._target import (
    _fit_and_resolve_estimator_target_spec,
    _resolve_estimator_target_spec,
)
from skactiveml.stream import VariableUncertainty
from skactiveml.stream._target import (
    _fit_and_resolve_clf_target_spec,
    _resolve_clf_target_spec,
)
from skactiveml.utils import MISSING_LABEL, TargetSpec


class _RecentWindowClassifier(SkactivemlClassifier):
    """Publish the target specification of the most recent samples only."""

    def fit(self, X, y, sample_weight=None):
        self.pwc_ = ParzenWindowClassifier(
            missing_label=self.missing_label
        ).fit(X[-4:], y[-4:])
        self.target_spec_ = self.pwc_.target_spec_
        self.classes_ = self.pwc_.classes_
        return self

    def predict_proba(self, X):
        return self.pwc_.predict_proba(X)


class TestResolveClfTargetSpec(unittest.TestCase):
    def setUp(self):
        self.qs = VariableUncertainty()

    def test_resolve_clf_target_spec(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, MISSING_LABEL])
        capability_error = "does not support target capability"

        fitted_clf = ParzenWindowClassifier(classes=[0, 1]).fit(X, y)
        self.assertIs(
            _resolve_clf_target_spec(self.qs, fitted_clf, y),
            fitted_clf.target_spec_,
        )

        prefitted_clf = SklearnClassifier(GaussianNB().fit(X, [0, 1, 0]))
        target_spec = _resolve_clf_target_spec(self.qs, prefitted_clf, None)
        self.assertEqual(target_spec.classes, (0, 1))
        self.assertIs(target_spec, prefitted_clf.target_spec_)

        unfitted_clf = ParzenWindowClassifier(classes=[0, 1])
        for labels in (y, None):
            target_spec = _resolve_clf_target_spec(
                self.qs, unfitted_clf, labels
            )
            self.assertEqual(target_spec.target_type, "single-output")
            self.assertEqual(target_spec.classes, (0, 1))
        self.assertFalse(hasattr(unfitted_clf, "target_spec_"))
        target_spec = _resolve_clf_target_spec(
            self.qs, ParzenWindowClassifier(), y
        )
        self.assertEqual(target_spec.classes, (0, 1))
        with self.assertRaisesRegex(ValueError, "No class label is observed"):
            _resolve_clf_target_spec(self.qs, ParzenWindowClassifier(), None)

        multilabel_clf = SklearnClassifier(
            MultiOutputClassifier(
                SGDClassifier(loss="log_loss", random_state=0)
            ),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        )
        for labels in (np.array([[0, 1], [1, 0], [-1, -1]]), None):
            with self.assertRaisesRegex(ValueError, capability_error):
                _resolve_clf_target_spec(self.qs, multilabel_clf, labels)

    def test_fitted_clf_capability_uses_training_specification(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, MISSING_LABEL])
        clf = AnnotatorEnsembleClassifier(
            estimators=[("pwc", ParzenWindowClassifier(classes=[0, 1]))],
            classes=[0, 1],
        ).fit(X, y[:, None])
        with patch.object(
            AnnotatorEnsembleClassifier,
            "_target_capabilities",
            new_callable=PropertyMock,
            return_value=frozenset(
                {("classification", "single-output", "single-annotator")}
            ),
        ):
            for labels in (y, None):
                with self.assertRaisesRegex(
                    ValueError, "AnnotatorEnsembleClassifier does not support"
                ):
                    _resolve_clf_target_spec(self.qs, clf, labels)


class TestClassifierTargetParity(unittest.TestCase):
    def test_classifier_rules_match_pool(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        y_int = np.array([0, 1, -1, -1])
        y_matrix = np.column_stack([y, y])
        y_multilabel = np.array(
            [[0, 1], [1, 0], [MISSING_LABEL] * 2, [MISSING_LABEL] * 2]
        )
        fitted_classifiers = {
            "pwc": ParzenWindowClassifier(classes=[0, 1]).fit(X, y),
            "sliding_window": SlidingWindowClassifier(
                ParzenWindowClassifier(classes=[0, 1])
            ).fit(X, y),
            "pwc_int": ParzenWindowClassifier(
                classes=[0, 1], missing_label=-1
            ).fit(X, y_int),
            "annotator_ensemble": AnnotatorEnsembleClassifier(
                estimators=[
                    ("pwc_0", ParzenWindowClassifier(classes=[0, 1])),
                    ("pwc_1", ParzenWindowClassifier(classes=[0, 1])),
                ],
                classes=[0, 1],
            ).fit(X, y_matrix),
            "multilabel": SklearnClassifier(
                MultiOutputClassifier(
                    SGDClassifier(loss="log_loss", random_state=0)
                ),
                classes=[[0, 1], [0, 1]],
                target_type="multi-label",
            ).fit(X, y_multilabel),
        }
        classifiers = [
            *((name, clf, True) for name, clf in fitted_classifiers.items()),
            *(
                (name, clone(clf), False)
                for name, clf in fitted_classifiers.items()
            ),
        ]
        label_cases = {
            "y": y,
            "y_column": y[:, None],
            "y_unlabeled": np.full(len(X), MISSING_LABEL),
            "y_int": y_int,
            "y_matrix": y_matrix,
            "y_multilabel": y_multilabel,
            "y_unknown_class": np.array([0, 2, MISSING_LABEL, MISSING_LABEL]),
            "y_inf": np.array([0, 1, np.inf, MISSING_LABEL]),
        }
        resolvers = (
            lambda clf, labels: _resolve_estimator_target_spec(
                ContrastiveAL(missing_label=clf.missing_label), clf, labels
            ),
            lambda clf, labels: _resolve_clf_target_spec(
                VariableUncertainty(), clf, labels
            ),
        )
        accepted = set()
        cases = product(classifiers, label_cases.items())
        for (name, clf, fitted), (label_name, labels) in cases:
            with self.subTest(clf=name, fitted=fitted, labels=label_name):
                outcomes = []
                for resolve in resolvers:
                    try:
                        outcomes.append(resolve(clf, labels))
                    except (TypeError, ValueError) as error:
                        outcomes.append(type(error))
                self.assertEqual(outcomes[0], outcomes[1])
                if isinstance(outcomes[0], TargetSpec):
                    accepted.add((name, fitted, label_name))
        self.assertEqual(
            accepted,
            {
                *(
                    (name, fitted, label_name)
                    for name in ("pwc", "sliding_window")
                    for fitted in (True, False)
                    for label_name in ("y", "y_column", "y_unlabeled")
                ),
                ("pwc_int", True, "y_int"),
                ("pwc_int", False, "y_int"),
                ("annotator_ensemble", True, "y"),
                ("annotator_ensemble", True, "y_column"),
                ("annotator_ensemble", True, "y_unlabeled"),
            },
        )

    def test_refit_of_lazily_resolved_classifier_matches_pool(self):
        X = np.random.RandomState(0).randn(30, 2)
        lr = LogisticRegression().fit(X, np.arange(30) % 3)
        y = np.full(len(X), MISSING_LABEL)
        y[:2] = [0, 1]
        clf = SklearnClassifier(FrozenEstimator(lr), classes=[0, 1, 2])
        results = [
            _fit_and_resolve_estimator_target_spec(
                UncertaintySampling(),
                clf,
                X,
                y,
                fit_estimator=True,
                sample_weight=None,
                estimator_name="clf",
                fit_name="fit_clf",
                estimator_types=[SkactivemlClassifier],
            ),
            _fit_and_resolve_clf_target_spec(
                VariableUncertainty(),
                clf,
                X,
                y,
                fit_clf=True,
                sample_weight=None,
            ),
        ]
        for fitted_clf, target_spec in results:
            self.assertEqual(target_spec.classes, (0, 1, 2))
            np.testing.assert_allclose(
                fitted_clf.predict_proba(X), lr.predict_proba(X)
            )

    def test_refit_rejects_labels_outside_fitted_classes(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([2, 2, 0, 1, 0, 1, MISSING_LABEL, MISSING_LABEL])
        clf = _RecentWindowClassifier()
        calls = [
            lambda: _fit_and_resolve_estimator_target_spec(
                UncertaintySampling(),
                clf,
                X,
                y,
                fit_estimator=True,
                sample_weight=None,
                estimator_name="clf",
                fit_name="fit_clf",
                estimator_types=[SkactivemlClassifier],
            ),
            lambda: _fit_and_resolve_clf_target_spec(
                VariableUncertainty(),
                clf,
                X,
                y,
                fit_clf=True,
                sample_weight=None,
            ),
        ]
        for call in calls:
            with self.assertRaisesRegex(
                ValueError, "labels outside `classes`"
            ):
                call()
