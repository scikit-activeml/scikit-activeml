import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler

from skactiveml import classifier
from skactiveml.utils import initialize_class_with_kwargs


class TestClassifier(unittest.TestCase):

    def test_classifiers(self):
        # Create data set for testing.
        self.X, self.y_true = load_breast_cancer(return_X_y=True)
        self.X = StandardScaler().fit_transform(self.X)
        self.y = np.repeat(self.y_true.reshape(-1, 1), 2, axis=1)
        self.classes = np.unique(self.y_true)
        self.missing_label = -1
        self.y[:100, 0] = self.missing_label
        self.y[200:, 0] = self.missing_label
        self.estimator = BaggingClassifier(random_state=1,
            base_estimator=GaussianProcessClassifier(random_state=1))
        #self.estimator = GaussianProcessClassifier(random_state=1)

        # Build dictionary of attributes.
        self.classifiers = {}
        for clf in classifier.__all__:
            self.classifiers[clf] = getattr(classifier, clf)

        # Test predictions of classifiers.
        for clf in self.classifiers:
            print(clf)
            self._test_classifier(clf)

    def _test_classifier(self, clf):
        # Test classifier without fitting.
        clf_mdl = initialize_class_with_kwargs(self.classifiers[clf],
                                               estimator=self.estimator,
                                               classes=self.classes,
                                               missing_label=
                                               self.missing_label,
                                               random_state=1)
        self.assertRaises(ValueError, clf_mdl.fit, X=[], y=[])
        score = clf_mdl.score(self.X, self.y_true)
        self.assertTrue(score > 0)
        if hasattr(clf_mdl, 'predict_proba'):
            P_exp = np.ones((len(self.X), len(self.classes))) \
                    / len(self.classes)
            P = clf_mdl.predict_proba(self.X)
            np.testing.assert_array_equal(P_exp, P)
        if hasattr(clf_mdl, 'predict_freq'):
            F_exp = np.zeros((len(self.X), len(self.classes)))
            F = clf_mdl.predict_freq(self.X)
            np.testing.assert_array_equal(F_exp, F)

        # Test classifier after fitting.
        self.assertTrue(
            clf_mdl.fit(self.X, self.y).score(self.X, self.y_true) > 0.8)
        print(clf_mdl.fit(self.X, self.y).score(self.X, self.y_true))
        if hasattr(clf_mdl, 'predict_proba'):
            P = clf_mdl.predict_proba(self.X)
            self.assertTrue(np.sum(P != 1 / len(self.classes)) > 0)
        if hasattr(clf_mdl, 'predict_freq'):
            F = clf_mdl.predict_freq(self.X)
            self.assertTrue(np.sum(F) > 0)

        # Training on data with only missing labels.
        clf_mdl = initialize_class_with_kwargs(self.classifiers[clf],
                                               estimator=self.estimator,
                                               classes=self.classes,
                                               missing_label=self.missing_label,
                                               random_state=1)
        clf_mdl.fit(X=self.X, y=np.ones_like(self.y) * -1)
        self.assertEqual(clf_mdl.score(self.X, self.y_true), score)
