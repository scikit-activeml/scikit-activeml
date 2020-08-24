import numpy as np
import unittest

from .._expected_error import ExpectedErrorReduction as EER
from ...classifier._new_pwc import PWC


class TestPWC(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.X_cand = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.clf = PWC(classes=[0, 1, 2])

    def test_init(self):
        self.assertRaises(TypeError, EER)
        self.assertRaises(TypeError, EER, clf=3, classes=[0, 1, 2])
        self.assertRaises(ValueError, EER, clf=self.clf, classes=[0, 1, 2], method='abc')
        self.assertRaises(ValueError, EER, clf=self.clf, classes=[0, 1, 2], C=np.ones((2, 3)))
        eer = EER(clf=self.clf, classes=[0, 1, 2], C=np.ones((2, 2)))
        self.assertIsNotNone(eer.classes_)

    def test_query(self):
        eer = EER(clf=self.clf, classes=[0, 1, 2], C=np.ones((2, 2)))
        self.assertRaises(ValueError, eer.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, eer.query, X_cand=[], X=self.X, y=self.y)
        self.assertRaises(ValueError, eer.query, X_cand=np.zeros((2, 2)), X=self.X, y=self.y)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y, classes=[0, 1, 3])
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y, method='abc')


if __name__ == '__main__':
    unittest.main()
