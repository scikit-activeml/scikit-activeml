import unittest

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

from skactiveml.base import SkactivemlConditionalEstimator
from skactiveml.regressor._wrapper import SklearnConditionalEstimator
from skactiveml.utils._approximation import conditional_expect


class TestApproximation(unittest.TestCase):
    def setUp(self):
        self.random_state = 0

    def test_conditional_expectation(self):

        cond_est = SklearnConditionalEstimator(estimator=GaussianProcessRegressor())
        X_train = np.array([[0, 2, 3], [1, 3, 4], [2, 4, 5], [3, 6, 7]])
        y_train = np.array([-1, 2, 1, 4])
        cond_est.fit(X_train, y_train)

        def dummy_func(y):
            return 0

        X = np.arange(2 * 3).reshape((2, 3))
        res = conditional_expect(X=X, func=dummy_func, cond_est=cond_est)
        np.array_equal(res, np.zeros((2, 3)))

        X_train = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5], [4, 5, 6]])
        cond_est.fit(X_train, y_train)
        X = np.arange(4 * 3).reshape((4, 3))
        res = conditional_expect(X=X, func=dummy_func, cond_est=cond_est)
        np.array_equal(res, np.zeros(4))

    def test_conditional_expectation_2(self):
        class DummyCondEst(SkactivemlConditionalEstimator):
            def fit(self, X, y, sample_weight=None):
                return self

            def estimate_conditional_distribution(self, X):
                return norm(loc=np.zeros(len(X)))

        cond_est = DummyCondEst()

        X = np.array([[0]])

        arg_dicts = [
            {"method": "assume_linear"},
            {"method": "quad", "quad_dict": {"epsabs": 0.1}},
            {
                "method": "quantile",
                "n_integration_samples": 300,
                "quantile_method": "simpson",
            },
            {"method": "monte_carlo", "n_integration_samples": 300},
        ]

        results = list()

        for arg_dict in arg_dicts:
            results.append(
                conditional_expect(
                    X,
                    lambda x: x**2,
                    cond_est=cond_est,
                    random_state=self.random_state,
                    **arg_dict
                )
            )

        print(results)

    def test_conditional_expectation_3(self):
        def diff(f):
            h = 2 ** (-5)
            return lambda x: (f(x + h) - f(x - h)) / (2 * h)

        dist = norm()

        y = np.linspace(0, 1, 1000)
        g = dist.pdf(dist.ppf(y))
        h = diff(dist.ppf)(y)
        f = g * h

        plt.plot(y, f)
        plt.show()

    def test_conditional_expectation_4(self):
        dist = norm()

        def f(x):
            return -dist.logpdf(x)

        alpha = np.linspace(0, 1, 1000)
        g = f(dist.ppf(alpha))

        plt.plot(alpha, g)
        plt.show()
