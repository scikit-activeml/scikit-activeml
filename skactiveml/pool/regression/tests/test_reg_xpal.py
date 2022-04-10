import unittest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from skactiveml.pool.regression._reg_xpal import RegxPal
from skactiveml.regressor._nwr import NWR
from skactiveml.regressor._wrapper import SklearnRegressor


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

        qs = RegxPal(random_state=0, n_monte_carlo_samples=100)

        # reg = SklearnRegressor(estimator=LinearRegression())
        reg = NWR(metric_dict={"gamma": 10.0})

        X_cand = np.linspace(-3, 6, 1000)
        X_cand_r = X_cand.reshape(-1, 1)
        X = np.array([[1], [2], [3]])
        E = np.append(X_cand_r, X, axis=0)
        y = np.array([1, 5, 3])
        reg.fit(X, y)

        query_indices, utilities = qs.query(
            X_cand.reshape(-1, 1),
            reg,
            E,
            X,
            y,
            batch_size=1,
            assume_linear=True,
            return_utilities=True,
        )

        y_pred = reg.predict(X_cand.reshape(-1, 1))

        plt.plot(X_cand, y_pred)
        plt.scatter(X, y, marker="x")
        plt.scatter(X_cand_r, np.ones_like(X_cand_r))
        plt.plot(X_cand, utilities)
        plt.show()

    def test_three(self):

        qs = RegxPal(
            random_state=0, n_monte_carlo_samples=100, metric_dict={"gamma": 10.0}
        )

        # reg = SklearnRegressor(estimator=LinearRegression())
        reg = NWR(metric_dict={"gamma": 10.0})

        X_cand = np.linspace(-1, 1, 100)
        y_cand = X_cand.copy()
        y_cand[:50] = y_cand[:50] + np.random.randn(50)
        lbd_idx = np.random.choice(np.arange(100), size=20)
        # lbd_idx = np.arange(10)
        X = X_cand[lbd_idx].reshape(-1, 1)
        y = y_cand[lbd_idx]
        E = X_cand.reshape(-1, 1)
        # E = np.append(E.ravel(), np.linspace(-0.25, 0.25, 100)).reshape(-1, 1)
        reg.fit(X, y)

        query_indices, utilities = qs.query(
            X_cand.reshape(-1, 1),
            reg,
            E,
            X,
            y,
            batch_size=1,
            assume_linear=True,
            return_utilities=True,
        )

        y_pred = reg.predict(X_cand.reshape(-1, 1))

        plt.plot(X_cand, y_pred, color="red")
        plt.scatter(X, y, marker="x")
        plt.plot(X_cand, y_cand, color="blue")
        plt.plot(X_cand, utilities * 1000, color="green")

        # notebook alle strategien, xpal 0 samples
        # fälle für parameter
        plt.show()
