import unittest
import numpy as np

from copy import deepcopy
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from skactiveml import classifier
from skactiveml.utils import initialize_class_with_kwargs


class TestClassifier(unittest.TestCase):

    def test_classifiers(self):
        # Create data set for testing.
        self.X, self.y_true = make_blobs(random_state=0)
        self.X = StandardScaler().fit_transform(self.X)
        self.y = np.repeat(self.y_true.reshape(-1, 1), 2, axis=1)
        self.y = self.y.astype('object')
        self.classes = np.unique(self.y_true)
        self.missing_label = None
        self.y[:100, 0] = self.missing_label
        self.y[200:, 1] = self.missing_label
        self.y_missing_label = np.full_like(self.y, self.missing_label)
        self.estimator = GaussianNB()
        pwc = classifier.CMM(missing_label=self.missing_label, random_state=0)
        gnb = classifier.SklearnClassifier(GaussianNB(),
                                           missing_label=self.missing_label)
        self.estimators = [('PWC', pwc), ('GaussianNB', gnb)]

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
                                               estimators=self.estimators,
                                               classes=self.classes,
                                               missing_label=
                                               self.missing_label,
                                               voting='soft',
                                               random_state=0)
        clf_mdl_copy = deepcopy(clf_mdl)
        clf_mdl_copy.classes = None
        if isinstance(clf_mdl_copy, classifier.MultiAnnotClassifier):
            clf_mdl_copy.estimators = [clf_mdl_copy.estimators[1]]
        check_estimator(clf_mdl_copy)
        self.assertRaises(ValueError, clf_mdl.fit, X=[], y=[])
        clf_mdl.fit(X=self.X, y=self.y_missing_label)
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
        print(clf_mdl.fit(self.X, self.y).score(self.X, self.y_true))
        self.assertTrue(
            clf_mdl.fit(self.X, self.y).score(self.X, self.y_true) > 0.8)
        if hasattr(clf_mdl, 'predict_proba'):
            P = clf_mdl.predict_proba(self.X)
            self.assertTrue(np.sum(P != 1 / len(self.classes)) > 0)
        if hasattr(clf_mdl, 'predict_freq'):
            F = clf_mdl.predict_freq(self.X)
            self.assertTrue(np.sum(F) > 0)

        # Training on data with only missing labels.
        clf_mdl = initialize_class_with_kwargs(self.classifiers[clf],
                                               estimator=self.estimator,
                                               estimators=self.estimators,
                                               classes=self.classes,
                                               missing_label=
                                               self.missing_label,
                                               voting='soft',
                                               random_state=0)
        clf_mdl.fit(X=self.X, y=self.y_missing_label)
        self.assertEqual(clf_mdl.score(self.X, self.y_true), score)
