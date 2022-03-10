import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import NotFittedError

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier.multiannotator import AnnotatorEnsembleClassifier
from skactiveml.utils import MISSING_LABEL


class TestAnnotatorEnsembleClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y_true = make_blobs(n_samples=300, random_state=0)
        self.y = np.array([self.y_true, self.y_true], dtype=float).T
        self.y[:100, 0] = MISSING_LABEL

    def test_init_param_estimators(self):
        clf = AnnotatorEnsembleClassifier(estimators="Test")
        self.assertEqual(clf.estimators, "Test")
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(estimators=None)
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(estimators=[("GNB", GaussianNB())])
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(
            estimators=[
                (
                    "ParzenWindowClassifier",
                    ParzenWindowClassifier(missing_label=0),
                )
            ]
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(
            estimators=[
                (
                    "ParzenWindowClassifier",
                    ParzenWindowClassifier(missing_label="a"),
                )
            ]
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(
            classes=[0, 1],
            estimators=[
                (
                    "ParzenWindowClassifier",
                    ParzenWindowClassifier(classes=[0, 2]),
                )
            ],
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(
            estimators=[
                (
                    "ParzenWindowClassifier",
                    ParzenWindowClassifier(classes=[0, 1]),
                )
            ]
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        perc = SklearnClassifier(Perceptron())
        clf = AnnotatorEnsembleClassifier(
            estimators=[("perc", perc)], voting="soft"
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)

    def test_init_param_voting(self):
        pwc = ParzenWindowClassifier()
        gnb = SklearnClassifier(GaussianNB())
        estimators = [("pwc", pwc), ("gnb", gnb)]
        clf = AnnotatorEnsembleClassifier(estimators=estimators, voting="Test")
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = AnnotatorEnsembleClassifier(estimators=estimators, voting=1)
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)

    def test_fit(self):
        pwc = ParzenWindowClassifier(classes=[1, 2])
        gnb = SklearnClassifier(GaussianNB(), classes=[1, 2])
        clf = AnnotatorEnsembleClassifier(
            estimators=[("ParzenWindowClassifier", pwc)], classes=[1, 2]
        )
        np.testing.assert_array_equal(clf.classes, gnb.classes)
        np.testing.assert_array_equal(clf.classes, pwc.classes)
        pwc = ParzenWindowClassifier(classes=np.arange(3))
        gnb = SklearnClassifier(GaussianNB(), classes=np.arange(3))
        clf = AnnotatorEnsembleClassifier(
            estimators=[("ParzenWindowClassifier", pwc), ("GNB", gnb)],
            voting="soft",
            classes=np.arange(3),
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y[:, 0])

    def test_predict_proba(self):
        pwc = ParzenWindowClassifier()
        gnb = SklearnClassifier(GaussianNB())
        clf = AnnotatorEnsembleClassifier(
            estimators=[("ParzenWindowClassifier", pwc), ("GNB", gnb)],
            voting="soft",
        )
        self.assertRaises(NotFittedError, clf.predict_proba, X=self.X)
        clf.fit(X=self.X, y=self.y)
        P = clf.predict_proba(X=self.X)
        np.testing.assert_allclose(np.ones(len(P)), P.sum(axis=1))
        clf.voting = "hard"
        clf.fit(X=self.X, y=self.y)
        P = clf.predict_proba(X=self.X)
        np.testing.assert_allclose(np.ones(len(P)), P.sum(axis=1))

    def test_predict(self):
        pwc = ParzenWindowClassifier(random_state=0)
        gnb = SklearnClassifier(GaussianNB(), random_state=0)
        clf = AnnotatorEnsembleClassifier(
            estimators=[("ParzenWindowClassifier", pwc), ("GNB", gnb)],
            voting="soft",
            random_state=0,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(X=self.X, y=self.y)
        y_pred_soft = clf.predict(X=self.X)
        self.assertEqual(len(y_pred_soft), len(self.X))
        self.assertTrue(clf.score(self.X, self.y_true), 0.8)
        clf.voting = "hard"
        clf.fit(X=self.X, y=self.y)
        y_pred_hard = clf.predict(X=self.X)
        self.assertEqual(len(y_pred_hard), len(self.X))
        self.assertTrue(clf.score(self.X, self.y_true), 0.8)
        clf.fit(X=self.X, y=self.y, sample_weight=np.ones_like(self.y))
        y_pred_hard = clf.predict(X=self.X)
        self.assertEqual(len(y_pred_hard), len(self.X))
        self.assertTrue(clf.score(self.X, self.y_true), 0.8)
