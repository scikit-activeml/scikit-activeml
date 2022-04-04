import unittest

import numpy as np
import matplotlib.pyplot as plt

from skactiveml.regressor.estimator._nichke import NormalInverseChiKernelEstimator


class TestNICHE(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 2, 3])

        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_estimate_posterior(self):
        nichke = NormalInverseChiKernelEstimator()
        nichke.fit(self.X, self.y)

        mu, std = nichke.predict(self.X_cand, return_std=True)
        print(mu, std)
