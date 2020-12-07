import numpy as np
import unittest

from skactiveml.pool import McPAL
from skactiveml.classifier import PWC
from skactiveml.utils import MISSING_LABEL


class TestMcPAL(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.utility_weight = np.ones(len(self.X)) / len(self.X)
        self.X_cand = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.C = np.eye(3)
        self.clf = PWC(classes=self.classes)

    # Test init parameters
    def test_init_param_clf(self):
        pal = McPAL(clf=3)
        self.assertTrue(hasattr(pal, 'clf'))
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y)

    def test_init_param_prior(self):
        pal = McPAL(self.clf, prior=0)
        self.assertTrue(hasattr(pal, 'prior'))
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y)

        pal = McPAL(self.clf, prior='string')
        self.assertTrue(hasattr(pal, 'prior'))
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y)

    def test_init_param_m_max(self):
        pal = McPAL(self.clf, m_max=-2)
        self.assertTrue(hasattr(pal, 'm_max'))
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y)

        pal = McPAL(self.clf, m_max=1.5)
        self.assertTrue(hasattr(pal, 'm_max'))
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y)

    def test_init_param_random_state(self):
        pal = McPAL(self.clf, random_state='string')
        self.assertTrue(hasattr(pal, 'random_state'))
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y)

    # Test query parameters
    def test_query_param_X_cand(self):
        pal = McPAL(self.clf)
        self.assertRaises(ValueError, pal.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, pal.query, X_cand=[], X=self.X, y=self.y)

    def test_query_param_X(self):
        pal = McPAL(self.clf)
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=np.ones((5, 3)), y=self.y)

    def test_query_param_y(self):
        pal = McPAL(self.clf)
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])

    def test_query_param_sample_weight(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, sample_weight='string')
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, sample_weight=np.ones(3))

    def test_query_param_utility_weight(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, utility_weight='string')
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, utility_weight=np.ones(3))

    def test_query_param_batch_size(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y,
                          batch_size=1.0)
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y,
                          batch_size=0)

    def test_query_param_return_utilities(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          return_utilities=None)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          return_utilities=[])
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          return_utilities=0)

    def test_query(self):
        mcpal = McPAL(clf=self.clf)
        self.assertRaises(ValueError, mcpal.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, mcpal.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, mcpal.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])

        # Test missing labels
        X_cand = [[0], [1], [2], [3]]
        mcpal = McPAL(clf=PWC(classes=[0, 1]))
        _, utilities = mcpal.query(X_cand, [[1]], [MISSING_LABEL],
                                   return_utilities=True)
        self.assertEqual(utilities.shape, (1, len(X_cand)))
        self.assertEqual(len(np.unique(utilities)), 1)

        _, utilities = mcpal.query(X_cand, X=[[0], [1], [2]],
                                   y=[0, 1, MISSING_LABEL],
                                   return_utilities=True)
        self.assertGreater(utilities[0, 2], utilities[0, 1])
        self.assertGreater(utilities[0, 2], utilities[0, 0])

        # Test scenario
        X_cand = [[0], [1], [2], [5]]
        mcpal = McPAL(clf=PWC(classes=[0, 1]))

        best_indices = mcpal.query(X_cand, X=[[1]], y=[0])
        np.testing.assert_array_equal(best_indices, np.array([3]))

        _, utilities = mcpal.query(X_cand, X=[[1]], y=[0],
                                   return_utilities=True)
        min_utilities = np.argmin(utilities)
        np.testing.assert_array_equal(min_utilities, np.array([1]))

        best_indices = mcpal.query(X_cand=[[0], [1], [2]], X=[[0], [2]],
                                   y=[0, 1])
        np.testing.assert_array_equal(best_indices, [1])


if __name__ == '__main__':
    unittest.main()
