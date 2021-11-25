import unittest

import numpy as np

from skactiveml.pool.regression._gsx import GSx


class TestGSx(unittest.TestCase):

    def setUp(self):
        pass

    def test_query(self):

        def euclidean_distance(A, B):
            diff = A.reshape((A.shape[0], 1, A.shape[1])) \
                   - B.reshape((1, B.shape[0], B.shape[1]))
            d = (np.sum(diff**2, axis=2))**(1/2)
            return d

        gsx = GSx(x_metric=euclidean_distance, random_state=0)

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])

        query_indices = gsx.query(X_cand, batch_size=2)
        print(query_indices)
