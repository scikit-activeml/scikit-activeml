import numpy as np
import unittest

from skactiveml.pool import McPAL
from skactiveml.classifier import PWC


class TestMCPAL(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.weights = np.ones(len(self.X)) / len(self.X)
        self.X_cand = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.C = np.eye(3)
        self.clf = PWC(classes=self.classes)

    def test_init(self):
        self.assertRaises(TypeError, McPAL)
        self.assertRaises(TypeError, McPAL, clf=3)
        self.assertRaises(ValueError, McPAL, clf=self.clf, m_max=-2)
        self.assertRaises(ValueError, McPAL, clf=self.clf, m_max=1.5)
        self.assertRaises(ValueError, McPAL, clf=self.clf, prior=0)

    def test_query(self):
        mcpal = McPAL(clf=self.clf)
        self.assertRaises(ValueError, mcpal.query, X_cand=[], X=[], y=[],
                          weights=[])
        self.assertRaises(ValueError, mcpal.query, X_cand=[], X=self.X,
                          y=self.y, weights=self.weights)
        self.assertRaises(ValueError, mcpal.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1], weights=self.weights)

    def test_scenario(self):
        X_cand = [[0], [1], [2], [5]]
        mcpal = McPAL(clf=PWC(classes=[0, 1]), classes=[0, 1], C=np.eye(2))

        best_indices = mcpal.query(X_cand, X=[[1]], y=[0],
                                   weights=[1, 1, 1, 1])
        np.testing.assert_array_equal(best_indices, np.array([3]))

        utilities = mcpal.query(X_cand, X=[[1]], y=[0], weights=[1, 1, 1, 1],
                                return_utilities=True)[1]
        min_utilities = np.argmin(utilities)
        np.testing.assert_array_equal(min_utilities, np.array([1]))

        best_indices = mcpal.query(X_cand=[[0], [1], [2]], X=[[0], [2]],
                                   y=[0, 1], weights=[1, 1, 1])
        np.testing.assert_array_equal(best_indices, [1])


if __name__ == '__main__':
    unittest.main()
