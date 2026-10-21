import unittest

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.stream import VariableUncertainty
from skactiveml.stream._target import _resolve_clf_target_spec
from skactiveml.utils import MISSING_LABEL


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
