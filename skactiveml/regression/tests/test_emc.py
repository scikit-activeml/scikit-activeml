import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from skactiveml.regression._emc import EMC
from skactiveml.regression._qbc import QBC
from skactiveml.regressor._wrapper import SklearnRegressor


class TestEMC(unittest.TestCase):

    def setUp(self):
        pass

    def test_query(self):
        qs = EMC(k_bootstraps=5, random_state=0)

        reg = SklearnRegressor(estimator=LinearRegression())

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        query_indices = qs.query(X_cand, reg, X, y, batch_size=2)
        print(query_indices)