import unittest
from itertools import product

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
        utils = np.array([4, 2, 5, 3, 1, 0], dtype=float)
        utils_copy = utils.copy()
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
        np.testing.assert_array_equal(utils, utils_copy)

        indices, batches = simple_batch(
            utils, random_state=42, batch_size=3, return_utilities=True
        )
        np.testing.assert_array_equal(indices[0:3], expected_indices[0:3])
        np.testing.assert_array_equal(batches[0:3], expected_batches[0:3])
        np.testing.assert_array_equal(utils, utils_copy)

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

    def test_combine_ranking_param_iter_ranking(self):
        for dtype in (bool, np.int64, np.uint64, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                actual = combine_ranking(np.array([0, 1, 1], dtype=dtype))
                self.assertEqual(actual.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(actual, [0.0, 1.0, 1.0])

        for first, second in product((list, tuple, np.array), repeat=2):
            for shape in [(4,), (2, 2)]:
                with self.subTest(first=first, second=second, shape=shape):
                    ranking_1 = first(np.array([1, 0, 1, 1]).reshape(shape))
                    ranking_2 = second(
                        np.array([100, 1, 99, 99]).reshape(shape)
                    )
                    actual = combine_ranking(ranking_1, ranking_2)
                    np.testing.assert_array_equal(
                        actual, np.array([3, 1, 2, 2]).reshape(shape)
                    )

        for rankings in [
            (),
            ([0, 1], [[0, 1]]),
            (np.array([0, 1]), np.array([[0, 1], [1, 2]])),
            ([0, 1], [0, 1, 2]),
            ([[0, 1], [1, 2]], [[0, 1, 2], [1, 2, 3]]),
            (["invalid", "ranking"],),
        ]:
            with self.subTest(rankings=rankings):
                with self.assertRaises(ValueError):
                    combine_ranking(*rankings)

    def test_combine_ranking_param_rank_method(self):
        rankings = ([2, 1, 1, 2, 1], [0, 100, 50, 0, 100])
        for method, expected in [
            (None, [3, 2, 1, 3, 2]),
            ("dense", [3, 2, 1, 3, 2]),
            ("min", [4, 2, 1, 4, 2]),
            ("max", [5, 3, 1, 5, 3]),
            ("average", [4.5, 2.5, 1, 4.5, 2.5]),
            ("ordinal", [4, 2, 1, 5, 3]),
        ]:
            with self.subTest(method=method):
                np.testing.assert_array_equal(
                    combine_ranking(*rankings, rank_method=method), expected
                )

        for method, error in [
            (0, TypeError),
            ([], TypeError),
            ("invalid", ValueError),
        ]:
            for values in [(np.array([0, 1]),), (np.array([np.nan]),) * 2]:
                with self.subTest(method=method, values=values):
                    with self.assertRaises(error):
                        combine_ranking(*values, rank_method=method)

    def test_combine_ranking_param_rank_per_batch(self):
        rankings = ([[2, 0, 0], [1, 2, 1]], [[0, 100, 101], [0, 0, -1]])
        for rank_per_batch, expected in [
            (False, [[5, 1, 2], [4, 5, 3]]),
            (True, [[3, 1, 2], [2, 3, 1]]),
        ]:
            with self.subTest(rank_per_batch=rank_per_batch):
                np.testing.assert_array_equal(
                    combine_ranking(*rankings, rank_per_batch=rank_per_batch),
                    expected,
                )

        for value in [0, 1.0, "True", None]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    combine_ranking(*rankings, rank_per_batch=value)

        ranking_1 = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [0, 0, 0, np.nan, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        ranking_2 = np.array(
            [[1, 0, 0, 100, np.nan], [1, 0, 0, 100, np.nan], [0, 0, 0, 0, 0]]
        )
        for method, partial, tied in [
            ("dense", [2, 1, 1, np.nan, np.nan], [1] * 5),
            ("min", [3, 1, 1, np.nan, np.nan], [1] * 5),
            ("max", [3, 2, 2, np.nan, np.nan], [5] * 5),
            ("average", [3, 1.5, 1.5, np.nan, np.nan], [3] * 5),
            ("ordinal", [3, 1, 2, np.nan, np.nan], [1, 2, 3, 4, 5]),
        ]:
            with self.subTest(method=method):
                actual = combine_ranking(
                    ranking_1,
                    ranking_2,
                    rank_per_batch=True,
                    rank_method=method,
                )
                np.testing.assert_array_equal(
                    actual, [[np.nan] * 5, partial, tied]
                )

    def test_combine_ranking_extreme_values(self):
        cases = [
            ([0, 0], [100, 101], [1, 2]),
            ([0, 1], [1000, -1000], [1, 2]),
            ([-1e308, 0, 1e308], [np.inf, 0, -np.inf], [1, 2, 3]),
            ([0, 0], [0, np.nextafter(0, 1)], [1, 2]),
            ([0, 0], [1, np.nextafter(1, np.inf)], [1, 2]),
            ([2**53 + 1, 2**53], [-np.inf, np.inf], [2, 1]),
            (
                np.array([2**64 - 1, 2**64 - 2], dtype=np.uint64),
                [0, 100],
                [2, 1],
            ),
            (
                [-np.inf, np.inf, -np.inf, np.inf],
                [100, 0, 101, -1],
                [1, 4, 2, 3],
            ),
        ]
        for ranking_1, ranking_2, expected in cases:
            with self.subTest(ranking_1=ranking_1, ranking_2=ranking_2):
                with np.errstate(all="raise"):
                    actual = combine_ranking(
                        np.asarray(ranking_1), np.asarray(ranking_2)
                    )
                np.testing.assert_array_equal(actual, expected)

    def test_combine_ranking_three_criteria(self):
        actual = combine_ranking(
            [1, 1, 1, 0, 1],
            [100, 100, 100, 1e308, 99],
            [1, 2, 2, 1e308, np.inf],
        )
        np.testing.assert_array_equal(actual, [3, 4, 4, 1, 2])

    def test_combine_ranking_broadcasting(self):
        ranking_1 = np.array([[2], [1]])
        ranking_2 = np.array([[100, 101, 100]])
        for rank_per_batch, expected in [
            (False, [[3, 4, 3], [1, 2, 1]]),
            (True, [[1, 2, 1], [1, 2, 1]]),
        ]:
            with self.subTest(rank_per_batch=rank_per_batch):
                np.testing.assert_array_equal(
                    combine_ranking(
                        ranking_1, ranking_2, rank_per_batch=rank_per_batch
                    ),
                    expected,
                )

        for shape in [(2, 4), (2, 2, 2)]:
            with self.subTest(shape=shape):
                actual = combine_ranking(
                    np.array([[0, 0, 1, 1], [10, 10, 0, 0]]).reshape(shape),
                    np.array([[101, 100, -100, 0], [1, 2, 1, 2]]).reshape(
                        shape
                    ),
                    rank_per_batch=True,
                )
                np.testing.assert_array_equal(
                    actual,
                    np.array([[2, 1, 3, 4], [3, 4, 1, 2]]).reshape(shape),
                )

    def test_combine_ranking_nan(self):
        for method, expected in [
            ("dense", [1, np.nan, 1, np.nan]),
            ("min", [1, np.nan, 1, np.nan]),
            ("max", [2, np.nan, 2, np.nan]),
            ("average", [1.5, np.nan, 1.5, np.nan]),
            ("ordinal", [1, np.nan, 2, np.nan]),
        ]:
            with self.subTest(method=method):
                actual = combine_ranking(
                    [0, np.nan, 0, 0], [2, 100, 2, np.nan], rank_method=method
                )
                np.testing.assert_array_equal(actual, expected)

        for rank_per_batch in [False, True]:
            with self.subTest(rank_per_batch=rank_per_batch):
                actual = combine_ranking(
                    [[np.nan, np.nan, np.nan], [0, 0, np.nan]],
                    [[1, np.nan, 3], [np.nan, 1, 0]],
                    rank_per_batch=rank_per_batch,
                )
                np.testing.assert_array_equal(
                    actual, [[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]]
                )
