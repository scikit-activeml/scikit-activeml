import numpy as np
import unittest

from sklearn.svm import SVR

from skactiveml.classifier import PWC
from skactiveml.pool import ALCE


class TestPWC(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.X_cand = np.zeros((5, 2))
        self.classes = [0, 1, 2]
        self.C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.regressor = SVR()
        self.pwc = PWC()

    def test_init(self):
        alce = ALCE(self.regressor, self.C)
        alce.query(self.X_cand, self.X, self.y)

        self.assertRaises(TypeError, ALCE)

        alce = ALCE(base_regressor=self.pwc, C=self.C)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

        alce = ALCE(base_regressor=self.regressor, C='A')
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y)

        alce = ALCE(base_regressor=self.regressor, C=self.C, embed_dim=1.5)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

        alce = ALCE(base_regressor=self.regressor, C=self.C, embed_dim=0)
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y)

    def test_query(self):
        alce = ALCE(base_regressor=self.regressor, C=self.C)
        alce.query(self.X_cand, self.X, self.y)
        self.assertRaises(ValueError, alce.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, alce.query, [], self.X, self.y)
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X,
                          y=[0, 1, 4, 0, 2, 1])
        self.assertRaises(ValueError, alce.query, X_cand=np.zeros((2, 3)),
                          X=self.X, y=self.y)

    def test_scenario(self):
        alce = ALCE(base_regressor=self.regressor, C=1-np.eye(2))
        query_indices = alce.query([[0], [100], [200]], [[0], [200]], [0, 1])
        np.testing.assert_array_equal(query_indices, [1])


if __name__ == '__main__':
    unittest.main()
