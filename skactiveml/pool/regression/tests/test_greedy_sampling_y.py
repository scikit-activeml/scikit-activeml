import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from skactiveml.pool.regression._greedy_sampling_y import GreedySamplingY
from skactiveml.regressor._wrapper import SklearnRegressor


class TestGSy(unittest.TestCase):
    def setUp(self):
        pass

    def test_query(self):
        gsy = GreedySamplingY(k_0=2, random_state=0)

        reg = SklearnRegressor(estimator=LinearRegression())

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        reg.fit(X, y)

        query_indices = gsy.query(X, y, candidates=X_cand, reg=reg, batch_size=2)
        print(query_indices)
