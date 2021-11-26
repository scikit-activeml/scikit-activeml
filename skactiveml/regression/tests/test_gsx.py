import unittest

import numpy as np

from skactiveml.regression._gsx import GSx


class TestGSx(unittest.TestCase):

    def setUp(self):
        pass

    def test_query(self):

        gsx = GSx(random_state=0)

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])

        query_indices = gsx.query(X_cand, batch_size=2)
        print(query_indices)
