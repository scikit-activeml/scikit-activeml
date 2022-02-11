import unittest

import numpy as np

from skactiveml.regressor.estimator._nwke import NormalInverseWishartKernelEstimator


class TestNWKE(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([[1, 0], [2, 0], [3, 0]])

        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_estimate_posterior(self):
        nwke = NormalInverseWishartKernelEstimator()
        nwke.fit(self.X, self.y)

        mu, cov = nwke.estimate_mu_cov(self.X_cand)
        print(mu, cov)
