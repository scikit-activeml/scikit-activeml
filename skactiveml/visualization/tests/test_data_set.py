import unittest

import numpy as np

from skactiveml.visualization import (
    gaussian_noise_generator_1d,
    sample_generator_1d,
)


class TestFeatureSpace(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(7).reshape(-1, 1)
        self.random_state = 0

    def test_gaussian_noise_generator_1d(self):
        default_std = 0
        noise_1 = gaussian_noise_generator_1d(
            self.X,
            (0, 3),
            default_std=default_std,
            random_state=self.random_state,
        )
        noise_2 = gaussian_noise_generator_1d(
            self.X,
            (0, 3),
            default_std=default_std,
            random_state=self.random_state,
        )
        self.assertEqual(noise_1.shape, (len(self.X),))
        self.assertTrue(np.all(noise_1[:3] != 0))
        self.assertTrue(np.all(noise_1[3:] == 0))
        np.testing.assert_array_equal(noise_1, noise_2)

        noise = gaussian_noise_generator_1d(
            self.X,
            default_std=1,
            random_state=self.random_state,
        )
        self.assertEqual(noise.shape, (len(self.X),))
        self.assertTrue(np.all(noise != 0))

    def test_sample_generator_1d(self):
        X_1 = sample_generator_1d(
            9, (0, 1, 2), (1, 2), random_state=self.random_state
        )
        X_2 = sample_generator_1d(
            9, (0, 1, 2), (1, 2), random_state=self.random_state
        )
        np.testing.assert_array_equal(X_1, X_2)
        self.assertEqual(X_1.shape, (9, 1))
        self.assertEqual(np.sum((0 <= X_1[:, 0]) & (X_1[:, 0] <= 1)), 6)
        self.assertEqual(np.sum((1 <= X_1[:, 0]) & (X_1[:, 0] <= 2)), 3)

        X = sample_generator_1d(
            10, (0, 1, 2), (1, 2), random_state=self.random_state
        )
        self.assertEqual(X.shape, (10, 1))
