import unittest

import numpy as np
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC

from skactiveml.regressor._wrapper import (
    SklearnRegressor,
    SklearnTargetDistributionRegressor,
)
from skactiveml.utils import MISSING_LABEL


class TestWrapper(unittest.TestCase):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.random_state = 0

    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 1, 1])

        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_params(self):
        reg = SklearnRegressor(estimator=SVC())
        self.assertRaises(TypeError, reg.fit, self.X, self.y)

    def test_fit_predict(self):
        estimator = LinearRegression()
        was_fitted_dict = {"is_fitted": False}

        def overwrite_fit(X_fit, y_fit):
            was_fitted_dict["is_fitted"] = True
            return

        estimator.fit = overwrite_fit
        reg = SklearnRegressor(estimator=estimator)
        y = np.full(3, MISSING_LABEL)
        reg.fit(self.X, y)
        self.assertFalse(was_fitted_dict["is_fitted"])

        reg_1 = SklearnRegressor(
            estimator=MLPRegressor(random_state=self.random_state),
            random_state=self.random_state,
        )

        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([3, 4, 1, 2, 1])

        reg_2 = clone(reg_1)
        sample_weight = np.arange(1, len(y) + 1)
        reg_1.fit(X, y, sample_weight=sample_weight)
        reg_2.fit(X, y)
        np.testing.assert_array_equal(reg_1.predict(X), reg_2.predict(X))

        reg_1 = SklearnRegressor(estimator=LinearRegression())
        reg_2 = clone(reg_1)
        reg_1.fit(X, y, sample_weight=sample_weight)
        reg_2.fit(X, y)
        self.assertTrue(np.any(reg_1.predict(X) != reg_2.predict(X)))


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
