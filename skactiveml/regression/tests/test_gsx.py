import unittest

import numpy as np

from skactiveml.regression._greedy_sampling_x import GSx


class TestGSx(unittest.TestCase):
    def setUp(self):
        pass

    def test_query(self):

        gsx = GSx(random_state=0)

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[-1 / 2, 1], [1, 0]])
        y = np.array([3 / 2, 4])

        query_indices = gsx.query(X, y, candidates=X_cand, batch_size=2)
        print(query_indices)
