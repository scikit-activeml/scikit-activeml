import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from skactiveml.regressor._nic_kernel_regressor import (
    NICKernelRegressor,
    NadarayaWatsonRegressor,
)
from skactiveml.utils import MISSING_LABEL
from scipy.stats import norm

from skactiveml.tests.template_estimator import TemplateProbabilisticRegressor


class TemplateTestNICKernelEstimator(TemplateProbabilisticRegressor):
    def setUp(
        self,
        estimator_class,
        init_default_params,
        fit_default_params=None,
        predict_default_params=None,
    ):
        super().setUp(
            estimator_class,
            init_default_params,
            fit_default_params,
            predict_default_params,
        )

    def test_init_param_metric(self):
        test_cases = []
        test_cases += [("rbf", None), (None, TypeError), ([], TypeError)]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self):
        test_cases = []
        test_cases += [
            ({"gamma": "mean"}, None),
            ("gamma", TypeError),
            ([], TypeError),
        ]
        self._test_param("init", "metric_dict", test_cases)

    def test_init_param_mu_0(self):
        if hasattr(self.estimator_class, "mu_0"):
            test_cases = []
            test_cases += [(0, None), (0.2, None), ("Test", TypeError)]
            self._test_param("init", "mu_0", test_cases)

    def test_init_param_kappa_0(self):
        if hasattr(self.estimator_class, "kappa_0"):
            test_cases = []
            test_cases += [
                (0.1, None),
                (1, None),
                (-1.0, None),
                ("Test", TypeError),
            ]
            self._test_param("init", "kappa_0", test_cases)

    def test_init_param_sigma_sq_0(self):
        if hasattr(self.estimator_class, "sigma_sq_0"):
            test_cases = []
            test_cases += [
                (1.0, None),
                (1, None),
                (-1.0, None),
                ("Test", TypeError),
            ]
            self._test_param("init", "sigma_sq_0", test_cases)

    def test_init_param_nu_0(self):
        if hasattr(self.estimator_class, "nu_0"):
            test_cases = []
            test_cases += [
                (2.5, None),
                (1, None),
                (-1.0, None),
                ("Test", TypeError),
            ]
            self._test_param("init", "nu_0", test_cases)

    def test_predict(self):
        reg = self.estimator_class(**self.start_parameter)
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y_missing = np.full(3, MISSING_LABEL)
        reg.fit(X, y_missing)
        y_return = reg.predict([[0, 0]])
        if self.estimator_class_string == "NadarayaWatsonRegressor":
            self.assertTrue(np.isnan(y_return[0]))
        else:
            self.assertEqual(y_return, 0)

        start_params = self.start_parameter
        if self.estimator_class_string == "NICKernelRegressor":
            start_params["kappa_0"] = 0
            start_params["nu_0"] = 2
        reg = self.estimator_class(**start_params)

        X = np.zeros((3, 1))
        y = np.arange(3)

        for i in range(3):
            w = np.array([0, 0, 0])
            w[i] = 1
            reg.fit(X, y, sample_weight=w)
            y_return = reg.predict([[0]])[0]
            self.assertEqual(y_return, y[i])

        X = np.zeros((500, 1))
        y = norm.rvs(loc=1.24, scale=0.0245, size=500, random_state=0)
        if self.estimator_class_string == "NICKernelRegressor":
            reg = self.estimator_class(kappa_0=0, nu_0=0)
            reg.fit(X, y)
            mu, sigma = reg.predict([[0]], return_std=True)
            np.testing.assert_almost_equal(mu, 1.24, decimal=3)
            np.testing.assert_almost_equal(sigma, 0.0245, decimal=3)

    def test_predict_target_distribution(self):
        # TODO: Test is missing
        pass


class TestNICKernelEstimator(
    TemplateTestNICKernelEstimator, unittest.TestCase
):
    def setUp(self):
        estimator_class = NICKernelRegressor
        self.estimator_class_string = "NICKernelRegressor"
        init_default_params = {
            "missing_label": MISSING_LABEL,
        }
        self.random_state = 0
        fit_default_params = {
            "X": np.zeros((3, 1)),
            "y": [0.5, 0.6, MISSING_LABEL],
        }
        predict_default_params = {"X": [[1]]}
        self.start_parameter = {
            "kappa_0": 1,
            "nu_0": 2,
            "mu_0": 0,
            "sigma_sq_0": 1,
            "metric": "rbf",
            "metric_dict": {"gamma": 10.0},
            "missing_label": MISSING_LABEL,
            "random_state": self.random_state,
        }
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 2, 3])
        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_fit(self):
        reg = NICKernelRegressor(**self.start_parameter)
        X = 5
        y = 7
        self.assertRaises(TypeError, reg.fit, X, y)

        w = np.zeros_like(self.y)
        self.assertRaises(ValueError, reg.fit, self.X, self.y, w)

    def test_missing_label(self):
        self.missing_label = -1
        start_params = self.start_parameter.copy()
        start_params["missing_label"] = -1
        reg_other_missing_label = NICKernelRegressor(**start_params)
        reg_usual_missing_label = NICKernelRegressor(**self.start_parameter)
        X = np.array([[0, 1], [0, 1], [1, 0], [0, 0]])
        y_other_missing_label = np.array([0.1, 0.2, -1, -1])
        y_usual_missing_label = np.array(
            [0.1, 0.2, MISSING_LABEL, MISSING_LABEL]
        )
        reg_other_missing_label.fit(X, y_other_missing_label)
        reg_usual_missing_label.fit(X, y_usual_missing_label)
        y_return_other = reg_other_missing_label.predict([[0, 0]])[0]
        y_return_usual = reg_usual_missing_label.predict([[0, 0]])[0]

        self.assertEqual(y_return_other, y_return_usual)

    def test_random_state(self):
        reg = NICKernelRegressor(**self.start_parameter)

        X_test = norm.rvs(size=(10, 2), random_state=self.random_state)
        X = norm.rvs(size=(8, 2), random_state=self.random_state)
        y = norm.rvs(size=8, random_state=self.random_state)

        reg.fit(X, y)
        prediction_1 = reg.predict(X_test)
        reg.fit(X, y)
        prediction_2 = reg.predict(X_test)

        np.testing.assert_almost_equal(prediction_1, prediction_2)


class TestNadarayaWatsonRegressor(
    TemplateTestNICKernelEstimator, unittest.TestCase
):
    def setUp(self):
        estimator_class = NadarayaWatsonRegressor
        self.estimator_class_string = "NadarayaWatsonRegressor"
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.random_state = 0
        init_default_params = {
            "metric": "rbf",
            "metric_dict": {"gamma": 10.0},
            "missing_label": MISSING_LABEL,
            "random_state": self.random_state,
        }
        self.start_parameter = init_default_params
        fit_default_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )

    def test_fit(self):
        reg = NadarayaWatsonRegressor(**self.init_default_params)
        self.assertRaises(NotFittedError, reg.predict, self.X)
        X = np.zeros((3, 1))
        y = np.arange(3)
        reg.fit(X, y)
        y_pred = reg.predict([[0]])[0]
        self.assertEqual(y_pred, np.average(y))
