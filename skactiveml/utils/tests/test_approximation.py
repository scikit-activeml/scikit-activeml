import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

from skactiveml.regressor._wrapper import SklearnConditionalEstimator
from skactiveml.utils._approximation import conditional_expect


class TestApproximation(unittest.TestCase):

    def test_conditional_expectation(self):

        cond_est = SklearnConditionalEstimator(
            estimator=GaussianProcessRegressor()
        )
        X_train = np.array([[0], [1], [2], [3]])
        y_train = np.array([-1, 2, 1, 4])
        cond_est.fit(X_train, y_train)

        def dummy_func(y):
            return 0

        X = np.arange(2 * 3 * 1).reshape((2, 3, 1))
        res = conditional_expect(X=X, func=dummy_func, cond_est=cond_est)
        np.array_equal(res, np.zeros((2, 3)))

        X_train = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5], [4, 5, 6]])
        cond_est.fit(X_train, y_train)
        X = np.arange(4 * 3 * 5).reshape((4, 3, 5))
        res = conditional_expect(X=X, func=dummy_func, cond_est=cond_est,
                                 axis=1)
        np.array_equal(res, np.zeros((4, 5)))
