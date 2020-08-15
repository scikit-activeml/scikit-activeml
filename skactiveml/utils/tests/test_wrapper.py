import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from skactiveml.utils._wrapper import SklearnClassifier


class TestClassifierWrapper(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((4, 1))
        self.y1 = ['tokyo', 'paris', 'nan', 'tokyo']
        self.y2 = ['tokyo', 'nan', 'nan', 'nan']

    def test_init(self):
        self.assertRaises(TypeError, SklearnClassifier, estimator=None)
        self.assertRaises(TypeError, SklearnClassifier, estimator=GaussianProcessRegressor())
        self.assertRaises(TypeError, SklearnClassifier, estimator=GaussianProcessClassifier(), missing_label=[2])
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), random_state=0)
        self.assertFalse(hasattr(clf, 'classes_'))
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), missing_label='nan', classes=['tokyo', 'paris'],
                                random_state=0)
        np.testing.assert_array_equal(['paris', 'tokyo'], clf.classes_)
        np.testing.assert_array_equal(['paris', 'tokyo'], clf._le._le.classes_)

    def test_fit(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier())
        self.assertRaises(ValueError, clf.fit, X=[], y=[])
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), classes=['tokyo', 'paris', 'new york'],
                                missing_label='nan').fit(X=[], y=[])
        self.assertFalse(clf.is_fitted_)
        clf.fit(self.X, self.y1)
        self.assertTrue(clf.is_fitted_)
        clf.fit(self.X, self.y2)
        self.assertFalse(clf.is_fitted_)

    def test_predict_proba(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), missing_label='nan')
        self.assertRaises(NotFittedError, clf.predict_proba, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        P = clf.predict_proba(X=self.X)
        est = GaussianProcessClassifier().fit(X=np.zeros((3, 1)), y=['tokyo', 'paris', 'tokyo'])
        P_exp = est.predict_proba(X=self.X)
        np.testing.assert_array_equal(P_exp, P)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.X, y=self.y2)
        P = clf.predict_proba(X=self.X)
        P_exp = np.ones((len(self.X), 1))
        np.testing.assert_array_equal(P_exp, P)
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), classes=['ny', 'paris', 'tokyo'],
                                missing_label='nan')
        P = clf.predict_proba(X=self.X)
        P_exp = np.ones((len(self.X), 3)) / 3
        np.testing.assert_array_equal(P_exp, P)
        clf.fit(X=self.X, y=self.y1)
        P = clf.predict_proba(X=self.X)
        P_exp = np.zeros((len(self.X), 3))
        P_exp[:, 1:] = est.predict_proba(X=self.X)
        np.testing.assert_array_equal(P_exp, P)

    def test_predict(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), missing_label='nan')
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(X=self.X, y=self.y1)
        y = clf.predict(X=self.X)
        est = GaussianProcessClassifier().fit(X=np.zeros((3, 1)), y=['tokyo', 'paris', 'tokyo'])
        y_exp = est.predict(X=self.X)
        np.testing.assert_array_equal(y, y_exp)
        np.testing.assert_array_equal(clf.classes_, est.classes_)
        clf.fit(X=self.X, y=self.y2)
        y = clf.predict(X=self.X)
        y_exp = ['tokyo'] * len(self.X)
        np.testing.assert_array_equal(y_exp, y)
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), classes=['ny', 'paris', 'tokyo'],
                                missing_label='nan', random_state=0)
        y = clf.predict(X=[[0]] * 1000)
        self.assertTrue(len(np.unique(y)) == len(clf.classes_))
        clf.fit(X=self.X, y=self.y1)


if __name__ == '__main__':
    unittest.main()
