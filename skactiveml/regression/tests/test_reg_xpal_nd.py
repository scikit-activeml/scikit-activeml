import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from skactiveml.regression._reg_xpal_nd import RegxPalNd
from skactiveml.regressor._wrapper import SklearnRegressor


class TestRegXPal(unittest.TestCase):

    def setUp(self):
        pass

    def test_query(self):
        qs = RegxPalNd(random_state=0)

        reg = SklearnRegressor(estimator=LinearRegression())

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[1, 2], [3, 4]])
        E = np.append(X_cand, X, axis=0)
        y = np.array([[0, 1], [1, 1]])
        reg.fit(X, y)

        query_indices = qs.query(X_cand, reg, E, X, y, batch_size=1,
                                 assume_linear=True)
        print(query_indices)
