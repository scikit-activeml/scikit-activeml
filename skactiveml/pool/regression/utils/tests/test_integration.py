import itertools
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from .....base import TargetDistributionEstimator
from ...utils._integration import conditional_expect
from .....regressor._wrapper import SklearnTargetDistributionRegressor


class TestApproximation(unittest.TestCase):
    def setUp(self):
        self.random_state = 0

    def test_conditional_expectation(self):

        cond_est = SklearnTargetDistributionRegressor(
            estimator=GaussianProcessRegressor()
        )
        X_train = np.array([[0, 2, 3], [1, 3, 4], [2, 4, 5], [3, 6, 7]])
        y_train = np.array([-1, 2, 1, 4])
        cond_est.fit(X_train, y_train)

        parameters_1 = [
            {"method": "assume_linear"},
            {"method": "monte_carlo", "n_integration_samples": 2},
            {"method": "monte_carlo", "n_integration_samples": 4},
            {"method": None, "quantile_method": "trapezoid"},
            {"method": "quantile", "quantile_method": "simpson"},
            {"method": "quantile", "quantile_method": "romberg"},
            {"method": "quantile", "quantile_method": "average"},
            {"method": "quantile", "quantile_method": "quadrature"},
            {"method": "quad"},
        ]

        parameters_2 = [
            {"vector_func": True},
            {"vector_func": False},
            {"vector_func": "optional"},
        ]

        X = np.arange(2 * 3).reshape((2, 3))

        for parameter_1, parameter_2 in itertools.product(parameters_1, parameters_2):
            parameter = parameter_1 | parameter_2

            def dummy_func(idx, x, y):
                if parameter["vector_func"] == "optional":
                    if parameter["method"] == "quad":
                        self.assertTrue(isinstance(idx, int))
                        self.assertTrue(isinstance(y, float))
                        self.assertEqual(x.shape, (3,))
                        return 0
                    else:
                        self.assertEqual(y.ndim, 2)
                        self.assertEqual(len(y), 2)
                        self.assertEqual(idx.dtype, int)
                        np.testing.assert_array_equal(x, X)
                        np.testing.assert_array_equal(idx, np.arange(2))
                        return np.zeros_like(y)
                elif parameter["vector_func"]:
                    self.assertEqual(y.ndim, 2)
                    self.assertEqual(len(y), 2)
                    self.assertEqual(idx.dtype, int)
                    np.testing.assert_array_equal(x, X)
                    np.testing.assert_array_equal(idx, np.arange(2))
                    return np.zeros_like(y)
                else:
                    self.assertTrue(isinstance(idx, int))
                    self.assertTrue(isinstance(y, float))
                    self.assertEqual(x.shape, (3,))
                    return 0

            res = conditional_expect(
                X=X,
                func=dummy_func,
                reg=cond_est,
                include_x=True,
                include_idx=True,
                **parameter
            )

            np.testing.assert_array_equal(res, np.zeros(2))

    def test_conditional_expectation_2(self):
        class DummyCondEst(TargetDistributionEstimator):
            def fit(self, X, y, sample_weight=None):
                return self

            def predict_target_distribution(self, X):
                return norm(loc=np.zeros(len(X)))

        cond_est = DummyCondEst()

        X = np.array([[0]])

        arg_dicts = [
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
                    reg=cond_est,
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

    def test_something(self):
        class DummyCondEst(TargetDistributionEstimator):
            def fit(self, X, y, sample_weight=None):
                return self

            def predict_target_distribution(self, X):
                return norm(loc=np.zeros(len(X)))

        X = (np.arange(11) - 6).reshape(-1, 1)
        dist = norm(loc=1)
        cond_est = DummyCondEst()
        expect_1 = conditional_expect(
            X,
            lambda x: -dist.logpdf(x),
            cond_est,
            n_integration_samples=7,
            method="gauss_hermite",
        )
        expect_2 = conditional_expect(
            X,
            lambda x: -dist.logpdf(x),
            cond_est,
            method="dynamic_quad",
            vector_func="both",
        )
        expect_3 = conditional_expect(
            X,
            lambda x: -dist.logpdf(x),
            cond_est,
            method="quantile",
            n_integration_samples=7,
            vector_func="both",
        )
        print(expect_1)
        print(expect_2)
        print(expect_3)
