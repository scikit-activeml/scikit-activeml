import unittest

import numpy as np

from skactiveml.utils import compute_vote_vectors
from skactiveml.utils._aggregation import majority_vote


class TestAggregation(unittest.TestCase):
    def test_majority_vote_preserves_labels_and_missing_rows(self):
        for classes in (
            np.array([3, 7], dtype=np.int64),
            np.array([3, 7], dtype=np.uint64),
        ):
            for missing in (np.nan, None, -2.5, -1):
                with self.subTest(classes=repr(classes), missing=missing):
                    y = np.empty((3, 3), dtype=object)
                    y[0] = [classes[0], classes[1], classes[0]]
                    y[1] = [classes[1], classes[1], missing]
                    y[2] = missing
                    result = majority_vote(
                        y, classes=classes, missing_label=missing
                    )
                    self.assertEqual(result[:2].tolist(), classes.tolist())
                    if missing is None:
                        self.assertIs(result[-1], None)
                    elif np.isnan(missing):
                        self.assertTrue(np.isnan(result[-1]))
                    else:
                        self.assertEqual(result[-1], missing)

    def test_compute_vote_vectors(self):
        y = [["tokyo", "paris", "tokyo"], ["paris", "paris", "nan"]]
        w = [[0.5, 1, 2], [0, 1, 0]]
        v_rec = compute_vote_vectors(
            y=y,
            w=w,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        v_exp = [[0, 1, 2.5], [0, 1, 0]]
        np.testing.assert_array_equal(v_rec, v_exp)

        y = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        w = [[0.5, 1, 2], [0, 1, 0]]
        v_rec = compute_vote_vectors(
            y=y, w=w, classes=[2, 4, 5], missing_label=np.nan
        )
        v_exp = [[0, 0, 0], [0, 0, 0]]
        np.testing.assert_array_equal(v_rec, v_exp)

        y = [[0, 0, 1], [1, np.nan, 1], [np.nan, np.nan, np.nan]]
        for w in (
            np.array([2, 3, 4]),
            np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        ):
            with self.subTest(weight_shape=w.shape):
                original_w = w.copy()
                v_rec = compute_vote_vectors(y, w=w, classes=[0, 1])
                np.testing.assert_array_equal(v_rec, [[4, 2], [0, 6], [0, 0]])
                np.testing.assert_array_equal(w, original_w)

    def test_compute_vote_vectors_no_label(self):
        y = np.full(shape=(2, 3), fill_value=np.nan)
        self.assertRaises(ValueError, compute_vote_vectors, y)

    def test_majority_vote(self):
        y = np.full(shape=(3, 3), fill_value=1, dtype=float)
        y[:, 1] = 0
        y[2, :] = np.nan

        y_aggregated_exp = np.full(shape=(3,), fill_value=1, dtype=float)
        y_aggregated_exp[2] = np.nan
        y_aggregated_rec = majority_vote(y=y)

        np.testing.assert_array_equal(y_aggregated_exp, y_aggregated_rec)

        with self.subTest(weights="sample-wise"):
            y_aggregated_rec = majority_vote(y=y, w=[2, 3, 4])
            np.testing.assert_array_equal(y_aggregated_exp, y_aggregated_rec)

        y_aggregated_rec = majority_vote(
            y=[[0, 1], [0, 1]], w=[[3, 1], [1, 3]]
        )
        np.testing.assert_array_equal(y_aggregated_rec, [0, 1])

    def test_majority_vote_no_label(self):
        y = np.full(shape=(3, 2), fill_value=np.nan)

        y_aggregate_exp = np.full(shape=(3,), fill_value=np.nan)
        y_aggregate_rec = majority_vote(y=y)

        np.testing.assert_array_equal(y_aggregate_exp, y_aggregate_rec)

        y = np.full(shape=(3, 2), fill_value=None, dtype=object)
        y_aggregate_rec = majority_vote(y=y, missing_label=None)

        self.assertEqual(y_aggregate_rec.dtype, object)
        self.assertEqual(y_aggregate_rec.tolist(), [None, None, None])

        for missing_label in (np.nan, None):
            for observed in (False, True):
                y = np.full((3, 2), missing_label)
                if observed:
                    y[0, 0] = 0
                for w in (
                    [1],
                    [1, 2],
                    np.ones((3, 1)),
                    np.ones((3, 3)),
                    np.ones((3, 2, 1)),
                ):
                    with self.subTest(
                        missing_label=missing_label,
                        observed=observed,
                        weight_shape=np.shape(w),
                    ):
                        with self.assertRaises(ValueError):
                            majority_vote(y, w=w, missing_label=missing_label)

    def test_weighted_majority_vote_encoding(self):
        y = [
            ["tokyo", "paris", "tokyo"],
            ["paris", "paris", "nan"],
            ["nan", "nan", "nan"],
        ]
        y_aggregated_rec = majority_vote(
            y=y, classes=["tokyo", "paris", "new york"], missing_label="nan"
        )
        y_aggregated_exp = ["tokyo", "paris", "nan"]

        np.testing.assert_array_equal(y_aggregated_rec, y_aggregated_exp)
