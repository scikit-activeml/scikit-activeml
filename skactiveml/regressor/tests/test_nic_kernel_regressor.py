import unittest

import numpy as np

from skactiveml.regressor._nic_kernel_regressor import NICKernelRegressor
from skactiveml.utils import MISSING_LABEL


class TestNICKernelEstimator(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 2, 3])
        self.X_cand = np.array([[2, 1], [3, 5]])
        self.random_state = 0
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

    def test_parameters(self):
        for param in ["kappa_0", "nu_0", "mu_0", "sigma_sq_0"]:
            start_params = self.start_parameter.copy()
            start_params[param] = "wrong_value"

            reg = NICKernelRegressor(**start_params)
            self.assertRaises(TypeError, reg.fit, self.X, self.y)

    def test_missing_label(self):
        self.missing_label = -1
        start_params = self.start_parameter.copy()
        start_params["missing_label"] = -1
        reg_other_missing_label = NICKernelRegressor(**start_params)
        reg_usual_missing_label = NICKernelRegressor(**self.start_parameter)
        X = np.array([[0, 1], [0, 1], [1, 0], [0, 0]])
        y_other_missing_label = np.array([0.1, 0.2, -1, -1])
        y_usual_missing_label = np.array([0.1, 0.2, MISSING_LABEL, MISSING_LABEL])
        reg_other_missing_label.fit(X, y_other_missing_label)
        reg_usual_missing_label.fit(X, y_usual_missing_label)
        y_return_other = reg_other_missing_label.predict([[0, 0]])[0]
        y_return_usual = reg_usual_missing_label.predict([[0, 0]])[0]

        self.assertEqual(y_return_other, y_return_usual)

    def test_fit(self):
        reg = NICKernelRegressor(**self.start_parameter)
        X = 5
        y = 7
        self.assertRaises(ValueError, reg.fit, X, y)

    def test_predict(self):
        reg = NICKernelRegressor(**self.start_parameter)
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y_missing = np.full(3, MISSING_LABEL)
        reg.fit(X, y_missing)
        y_return = reg.predict([[0, 0]])
        self.assertEqual(y_return, 0)

        start_params = self.start_parameter
        start_params["kappa_0"] = 0
        start_params["nu_0"] = 2
        reg = NICKernelRegressor(**start_params)

        X = np.zeros((3, 1))
        y = np.arange(3)

        for i in range(3):
            w = np.array([0, 0, 0])
            w[i] = 1
            reg.fit(X, y, sample_weight=w)
            y_return = reg.predict([[0]])[0]
            self.assertEqual(y_return, y[i])
