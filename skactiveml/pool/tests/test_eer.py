import numpy as np
import unittest

from skactiveml.pool import ExpectedErrorReduction as EER
from skactiveml.classifier import PWC


class TestPWC(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.X_cand = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.C = np.eye(3)
        self.clf = PWC(classes=self.classes)

    def test_init(self):
        self.assertRaises(TypeError, EER)
        self.assertRaises(TypeError, EER, clf=3, classes=self.classes)
        self.assertRaises(ValueError, EER, clf=self.clf, classes=self.classes, method='wrong_method')
        self.assertRaises(ValueError, EER, clf=self.clf, classes=self.classes, C=np.ones((2, 3)))
        self.assertRaises(ValueError, EER, clf=self.clf, classes=self.classes, C=np.ones((2, 2)))
        eer = EER(clf=self.clf, classes=self.classes, C=self.C)
        self.assertIsNotNone(eer.classes_)

    def test_query(self):
        eer = EER(clf=self.clf, classes=self.classes, C=self.C)
        self.assertRaises(ValueError, eer.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, eer.query, X_cand=[], X=self.X, y=self.y)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=[0, 1, 4, 0, 2, 1])

    def test_scenario(self):
        X_cand = [[0], [1], [2], [5]]
        eer = EER(clf=PWC(classes=[0,1]), classes=[0, 1], C=np.eye(2))
        np.testing.assert_array_equal(eer.query(X_cand, [[1]], [0], return_utilities=True)[1], np.array([[-1]*len(X_cand)]) * len(X_cand))
        np.testing.assert_array_equal(eer.query([[0], [1], [2]], [[0], [2]], [0, 1]), [1])


if __name__ == '__main__':
    unittest.main()
