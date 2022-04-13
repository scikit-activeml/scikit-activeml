import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

from skactiveml.regressor._wrapper import (
    SklearnRegressor,
    SklearnTargetDistributionRegressor,
)


class TestWrapper(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 1, 1])

        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_predict(self):
        reg = SklearnRegressor(estimator=LinearRegression())
        reg.fit(self.X, self.y)

        y_pred = reg.predict(self.X_cand)

        print(y_pred)


class TestCondEstWrapper(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 2, 3])

        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_estimate_cond(self):
        reg = SklearnTargetDistributionRegressor(estimator=GaussianProcessRegressor())
        reg.fit(self.X, self.y)

        y_pred = reg.predict(self.X_cand, return_std=True)

        print(y_pred)
