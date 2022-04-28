import unittest

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from skactiveml.pool.regression import QueryByCommittee
from skactiveml.regressor import NICKernelRegressor
from skactiveml.regressor._wrapper import SklearnRegressor


class TestQBC(unittest.TestCase):
    def setUp(self):
        pass

    def test_query(self):
        gsy = QueryByCommittee(random_state=0)

        reg = NICKernelRegressor()

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[1, 2], [3, 6], [5, 4], [7, 8]])
        y = np.array([0, 1, 2, 3])

        query_indices = gsy.query(X, y, candidates=X_cand, ensemble=reg, batch_size=2)
        print(query_indices)

    def test_dt(self):
        dist = norm(loc=1)
        a = dist.ppf(0.7)
        a = dist.moment(5)
