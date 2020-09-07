import numpy as np
import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from skactiveml import classifier


def initialize_class(class_obj, **kwargs):
    parameters = class_obj.__init__.__code__.co_varnames
    kwargs = dict(filter(lambda e: e[0] in parameters, kwargs.items()))
    return class_obj(**kwargs)


class TestClassifier(unittest.TestCase):

    def test_classifiers(self):
        # create data set
        X, y_true = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = np.repeat(y_true.reshape(-1, 1), 2, axis=1)
        classes = np.unique(y_true)
        missing_label = -1
        y[:100, 0] = missing_label
        y[200:, 0] = missing_label

        # build dictionary of attributes
        classifiers = {}
        for clf in classifier.__all__:
            classifiers[clf] = getattr(classifier, clf)

        # test predictions of classifiers
        for clf in classifiers:
            print(clf)
            clf_mdl = initialize_class(classifiers[clf],
                                       estimator=GaussianProcessClassifier(),
                                       classes=classes,
                                       missing_label=missing_label,
                                       random_state=1)
            self.assertRaises(ValueError, clf_mdl.fit, X=[], y=[])
            self.assertTrue(clf_mdl.score(X, y_true) > 0)
            if hasattr(clf_mdl, 'predict_proba'):
                P_exp = np.ones((len(X), len(classes))) / len(classes)
                P = clf_mdl.predict_proba(X)
                np.testing.assert_array_equal(P_exp, P)
            if hasattr(clf_mdl, 'predict_freq'):
                F_exp = np.zeros((len(X), len(classes)))
                F = clf_mdl.predict_freq(X)
                np.testing.assert_array_equal(F_exp, F)
            self.assertTrue(clf_mdl.fit(X, y).score(X, y_true) > 0)
            if hasattr(clf_mdl, 'predict_proba'):
                P = clf_mdl.predict_proba(X)
                self.assertTrue(np.sum(P != 1 / len(classes)) > 0)
            if hasattr(clf_mdl, 'predict_freq'):
                F = clf_mdl.predict_freq(X)
                self.assertTrue(np.sum(F) > 0)
