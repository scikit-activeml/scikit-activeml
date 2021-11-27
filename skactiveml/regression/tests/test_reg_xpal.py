import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from skactiveml.regression._gsy import GSy
from skactiveml.regression._reg_xpal import RegxPal
from skactiveml.regressor._nwr import NWR
from skactiveml.regressor._wrapper import SklearnRegressor

import matplotlib.pyplot as plt


class TestRegXPal(unittest.TestCase):

    def setUp(self):
        pass

    def test_query(self):
        qs = RegxPal(random_state=0)

        reg = SklearnRegressor(estimator=LinearRegression())

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[1, 2], [3, 4]])
        E = np.append(X_cand, X, axis=0)
        y = np.array([0, 1])
        reg.fit(X, y)

        query_indices = qs.query(X_cand, reg, E, X, y, batch_size=1)
        print(query_indices)

    def test_two(self):

        qs = RegxPal(random_state=0)

        reg = SklearnRegressor(estimator=LinearRegression())

        X_cand = np.linspace(-3, 6, 100)
        X_cand_r = np.array([[0], [-1], [4], [6]])
        X = np.array([[1], [2], [3]])
        E = np.append(X_cand_r, X, axis=0)
        y = np.array([1, 5, 3])
        reg.fit(X, y)

        query_indices, utilities = qs.query(X_cand.reshape(-1, 1), reg, E, X, y, batch_size=1,
                                            assume_linear=True, return_utilities=True)

        y_pred = reg.predict(X_cand.reshape(-1, 1))

        plt.plot(X_cand, y_pred)
        plt.plot(X_cand, utilities)
        plt.show()
