import numpy as np
import unittest

from skactiveml.classifier import MultiAnnotClassifier, PWC, \
    SklearnClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
from sklearn.utils.validation import NotFittedError


class TestMultiAnnotClassifier(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_blobs(n_samples=300, random_state=0)
        self.y = np.array([self.y, self.y], dtype=float).T
        self.y[:100, 0] = np.nan

    def test_init(self):
        clf = MultiAnnotClassifier(estimators='Test', missing_label=-1)
        self.assertEqual(clf.missing_label, -1)
        self.assertEqual(clf.classes, None)
        self.assertEqual(clf.estimators, 'Test')
        self.assertEqual(clf.voting, 'hard')
        self.assertEqual(clf.random_state, None)
        self.assertEqual(clf.cost_matrix, None)

    def test_fit(self):
        clf = MultiAnnotClassifier(estimators=None)
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = MultiAnnotClassifier(estimators=[('GNB', GaussianNB())])
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)
        clf = MultiAnnotClassifier(estimators=[('PWC', PWC(missing_label=0))])
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = MultiAnnotClassifier(estimators=[('PWC', PWC())], voting='test')
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y[:, 1])
        clf = MultiAnnotClassifier(estimators=
                                   [('PWC', PWC(missing_label='a'))])
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = MultiAnnotClassifier(classes=[0, 1],
                                   estimators=[('PWC', PWC(classes=[0, 2]))])
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        clf = MultiAnnotClassifier(estimators=[('PWC', PWC(classes=[0, 1]))])
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        perc = SklearnClassifier(Perceptron())
        clf = MultiAnnotClassifier(estimators=[('perc', perc)], voting='soft')
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)
        pwc = PWC(classes=[1, 2])
        gnb = SklearnClassifier(GaussianNB(), classes=[1, 2])
        clf = MultiAnnotClassifier(estimators=[('PWC', pwc)],
                                   classes=[1, 2])
        np.testing.assert_array_equal(clf.classes, gnb.classes)
        np.testing.assert_array_equal(clf.classes, pwc.classes)
        pwc = PWC(classes=np.arange(3))
        gnb = SklearnClassifier(GaussianNB(), classes=np.arange(3))
        clf = MultiAnnotClassifier(estimators=[('PWC', pwc), ('GNB', gnb)],
                                   voting='soft', classes=np.arange(3))
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y[:, 0])
        clf.fit(X=self.X, y=self.y)

    def test_predict_proba(self):
        pwc = PWC()
        gnb = SklearnClassifier(GaussianNB())
        clf = MultiAnnotClassifier(estimators=[('PWC', pwc), ('GNB', gnb)],
                                   voting='soft')
        self.assertRaises(NotFittedError, clf.predict_proba, X=self.X)
        clf.fit(X=self.X, y=self.y)
        P = clf.predict_proba(X=self.X)
        np.testing.assert_allclose(np.ones(len(P)), P.sum(axis=1))
        clf.voting = 'hard'
        clf.fit(X=self.X, y=self.y)
        P = clf.predict_proba(X=self.X)
        np.testing.assert_allclose(np.ones(len(P)), P.sum(axis=1))


if __name__ == '__main__':
    unittest.main()
