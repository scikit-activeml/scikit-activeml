import unittest

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from skactiveml.utils import rand_argmin, rand_argmax, simple_batch
from skactiveml.utils._selection import combine_ranking


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.a = [2, 5, 6, -1]
        self.b = [[2, 1], [3, 5]]
        self.c = [2, 2, 1, 1]
        self.d = [[2, 2], [1, 1]]
        self.e = [np.nan, 1]

    def test_rand_argmin(self):
        np.testing.assert_array_equal([3], rand_argmin(self.a))
        np.testing.assert_array_equal([1, 0], rand_argmin(self.b, axis=1))
        np.testing.assert_array_equal([0, 1], rand_argmin(self.b))
        np.testing.assert_array_equal(
            [2], rand_argmin(self.c, random_state=42)
        )
        np.testing.assert_array_equal(
            [1, 0], rand_argmin(self.d, axis=1, random_state=42)
        )
        np.testing.assert_array_equal(
            [1, 0], rand_argmin(self.d, random_state=42)
        )
        np.testing.assert_array_equal([3], rand_argmin(self.c, random_state=1))
        np.testing.assert_array_equal(
            [1, 1], rand_argmin(self.d, axis=1, random_state=1)
        )
        np.testing.assert_array_equal(
            [1, 1], rand_argmin(self.d, random_state=1)
        )
        np.testing.assert_array_equal([1], rand_argmin(self.e))

    def test_rand_argmax(self):
        np.testing.assert_array_equal([2], rand_argmax(self.a))
        np.testing.assert_array_equal([0, 1], rand_argmax(self.b, axis=1))
        np.testing.assert_array_equal([1, 1], rand_argmax(self.b))
        np.testing.assert_array_equal(
            [1], rand_argmax(self.c, random_state=42)
        )
        np.testing.assert_array_equal(
            [1, 0], rand_argmax(self.d, axis=1, random_state=42)
        )
        np.testing.assert_array_equal(
            [0, 1], rand_argmax(self.d, random_state=42)
        )
        np.testing.assert_array_equal(
            [0], rand_argmax(self.c, random_state=10)
        )
        np.testing.assert_array_equal(
            [1, 1], rand_argmax(self.d, axis=1, random_state=1)
        )
        np.testing.assert_array_equal(
            [0, 0], rand_argmax(self.d, random_state=10)
        )
        np.testing.assert_array_equal([1], rand_argmax(self.e))

    def test_simple_batch(self):
        utils = [4, 2, 5, 3, 1, 0]
        expected_indices = np.array([2, 0, 3, 1, 4, 5])
        expected_batches = np.array(
            [
                [4, 2, 5, 3, 1, 0],
                [4, 2, np.nan, 3, 1, 0],
                [np.nan, 2, np.nan, 3, 1, 0],
                [np.nan, 2, np.nan, np.nan, 1, 0],
                [np.nan, np.nan, np.nan, np.nan, 1, 0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 0],
            ]
        )
        self.assertRaises(
            TypeError,
            simple_batch,
            utils,
            random_state=42,
            batch_size="invalid",
        )
        self.assertRaises(
            ValueError, simple_batch, utils, random_state=42, batch_size=0
        )
        indices, batches = simple_batch(
            utils,
            random_state=42,
            batch_size=len(utils) + 1,
            return_utilities=True,
        )
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(batches, expected_batches)

        indices, batches = simple_batch(
            utils, random_state=42, batch_size=3, return_utilities=True
        )
        np.testing.assert_array_equal(indices[0:3], expected_indices[0:3])
        np.testing.assert_array_equal(batches[0:3], expected_batches[0:3])

        indices, batches = simple_batch(
            [[np.nan, np.nan], [np.nan, np.nan]],
            random_state=42,
            batch_size=1,
            return_utilities=True,
        )
        np.testing.assert_equal((0, 2), indices.shape)
        np.testing.assert_array_equal((0, 2, 2), batches.shape)

        indices = simple_batch(
            [[np.nan, np.nan], [np.nan, np.nan]],
            random_state=42,
            batch_size=1,
            return_utilities=False,
        )
        np.testing.assert_equal((0, 2), indices.shape)

        batch_size = 10
        idx, utils = simple_batch(
            np.arange(100),
            batch_size=batch_size,
            return_utilities=True,
            method="proportional",
        )
        self.assertEqual(batch_size, len(idx))
        self.assertEqual(
            np.sum(np.isnan(utils)), np.sum(np.arange(batch_size))
        )
        for i in range(batch_size):
            np.testing.assert_array_equal(
                np.argwhere(np.isnan(utils[i])).flatten(), np.sort(idx[:i])
            )

        # test proportional method
        N = 1000
        X = np.linspace(-5, 10, N)
        true_dens = 0.3 * norm(0, 1).pdf(X.reshape(-1, 1)) + 0.7 * norm(
            5, 1
        ).pdf(X.reshape(-1, 1))
        true_dens += np.mean(true_dens)
        true_dens = true_dens / np.sum(true_dens)
        sel = np.empty(100000)
        for i in range(100000):
            sel[i] = X[
                simple_batch(
                    true_dens[:, 0], method="proportional", random_state=i
                )[0]
            ]
        density = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(
            sel.reshape(-1, 1)
        )
        est_dens = np.exp(density.score_samples(X.reshape(-1, 1))).flatten()
        est_dens /= np.sum(est_dens)
        np.testing.assert_allclose(
            true_dens.flatten()[20:-20], est_dens[20:-20], rtol=0.1
        )

        self.assertRaises(ValueError, simple_batch, utils, method="string")

    def test_combine_ranking(self):
        self.assertRaises(
            ValueError,
            combine_ranking,
            np.array([0, 1]),
            np.array([[0, 1], [1, 2]]),
        )

        ranking_1 = np.array([0, 1, 1])
        ranking_2 = np.array([0.1, 0.2, 0.4])
        new_ranking = combine_ranking(ranking_2)
        np.testing.assert_array_equal(ranking_2, new_ranking)

        com_ranking = combine_ranking(ranking_1, ranking_2)

        self.assertTrue(com_ranking[1] > com_ranking[0])
        self.assertTrue(com_ranking[2] > com_ranking[1])

        ranking_1 = np.array([0.1, 0.1, 0.2])
        ranking_2 = np.array([14, 13, 12])
        com_ranking = combine_ranking(ranking_1, ranking_2)
        self.assertTrue(com_ranking[2] > com_ranking[0])
        self.assertTrue(com_ranking[0] > com_ranking[1])

        ranking_1 = np.array([[2, 3, 4], [1, 0, 0]])
        ranking_2 = np.array([[4, 3, 3], [4.5, 4, 10]])

        com_ranking = combine_ranking(
            ranking_1, ranking_2, rank_per_batch=True
        )

        self.assertTrue(com_ranking[0, 2] > com_ranking[0, 1])
        self.assertTrue(com_ranking[0, 1] > com_ranking[0, 0])

        self.assertTrue(com_ranking[1, 0] > com_ranking[1, 2])
        self.assertTrue(com_ranking[1, 2] > com_ranking[1, 1])
