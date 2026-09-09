import unittest

import numpy as np

from skactiveml.utils._label import _observed_numerical_labels
from skactiveml.utils import (
    is_labeled,
    is_unlabeled,
    labeled_indices,
    unlabeled_indices,
    check_missing_label,
    check_equal_missing_label,
)


class TestLabel(unittest.TestCase):
    def setUp(self):
        self.y1 = [np.nan, 2, 5, 10, np.nan]
        self.y2 = [np.nan, "2", "5", "10", np.nan]
        self.y3 = [None, 2, 5, 10, None]
        self.y4 = [None, "2", "5", "10", None]
        self.y5 = [8, -1, 1, 5, 2]
        self.y6 = ["paris", "france", "tokyo", "nan"]
        self.y7 = ["paris", "france", "tokyo", -1]
        self.y8 = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [-1, -1, -1]]
        self.y9 = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [-1, 0, -1]]

    def test_missing_masks_preserve_object_integer_targets(self):
        for missing in (np.nan, None, -2.5, -1):
            with self.subTest(missing=missing):
                y = np.array(
                    [[2**64 - 2, 2**64 - 1], [missing, missing]], dtype=object
                )
                np.testing.assert_array_equal(
                    is_unlabeled(y, missing), [[False, False], [True, True]]
                )
                np.testing.assert_array_equal(
                    is_labeled(y, missing, target_type="multi-label"),
                    [True, False],
                )
                self.assertEqual(y[0].tolist(), [2**64 - 2, 2**64 - 1])

    def test_is_unlabeled(self):
        self.assertRaises(ValueError, is_unlabeled, y=[0], target_type="auto")
        self.assertRaises(
            ValueError, is_unlabeled, y=[0], target_type="multi-output"
        )
        self.assertRaises(
            ValueError, is_unlabeled, y=[], target_type="multi-label"
        )
        np.testing.assert_array_equal(
            np.array([], dtype=bool),
            is_unlabeled(np.empty((0, 2)), target_type="multi-label"),
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y1, missing_label="2"
        )
        self.assertRaises(ValueError, is_unlabeled, [[]], missing_label="2")
        self.assertRaises(
            ValueError,
            is_unlabeled,
            y=np.zeros((1, 1, 1)),
            missing_label=-1,
        )
        self.assertRaises(
            ValueError,
            is_unlabeled,
            y=[0, 1],
            missing_label=-1,
            target_type="multi-label",
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y2, missing_label=np.nan
        )
        self.assertRaises(
            ValueError, is_unlabeled, y=self.y2, missing_label="2"
        )
        self.assertRaises(
            ValueError, is_unlabeled, y=self.y2, missing_label=None
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y3, missing_label="2"
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y3, missing_label=np.nan
        )
        self.assertRaises(TypeError, is_unlabeled, y=self.y4, missing_label=2)
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y4, missing_label="2"
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y5, missing_label="2"
        )
        self.assertRaises(TypeError, is_unlabeled, y=self.y6, missing_label=2)
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y6, missing_label=np.nan
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y7, missing_label=np.nan
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y7, missing_label=None
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y7, missing_label="2"
        )
        self.assertRaises(TypeError, is_unlabeled, y=self.y7, missing_label=-1)
        self.assertRaises(
            ValueError,
            is_unlabeled,
            y=self.y9,
            missing_label=-1,
            target_type="multi-label",
        )

        np.testing.assert_array_equal(
            np.array([], dtype=bool), is_unlabeled([])
        )
        np.testing.assert_array_equal(
            np.array([1, 0, 0, 0, 1], dtype=bool), is_unlabeled(self.y1)
        )
        np.testing.assert_array_equal(
            np.array([1, 0, 0, 0, 1], dtype=bool),
            is_unlabeled(self.y3, missing_label=None),
        )
        np.testing.assert_array_equal(
            np.array([1, 0, 0, 0, 1], dtype=bool),
            is_unlabeled(self.y4, missing_label=None),
        )
        np.testing.assert_array_equal(
            np.array([0, 0, 0, 0, 0], dtype=bool),
            is_unlabeled(self.y5, missing_label=None),
        )
        np.testing.assert_array_equal(
            np.array([0, 0, 0, 0, 0], dtype=bool),
            is_unlabeled(self.y5, missing_label=np.nan),
        )
        np.testing.assert_array_equal(
            np.array([0, 1, 0, 0, 0], dtype=bool),
            is_unlabeled(self.y5, missing_label=-1),
        )
        np.testing.assert_array_equal(
            np.array([0, 0, 0, 0], dtype=bool),
            is_unlabeled(self.y6, missing_label=None),
        )
        np.testing.assert_array_equal(
            np.array([0, 0, 0, 1], dtype=bool),
            is_unlabeled(self.y6, missing_label="nan"),
        )
        np.testing.assert_array_equal(
            np.array([0, 0, 0, 1], dtype=bool),
            is_unlabeled(self.y8, missing_label=-1, target_type="multi-label"),
        )
        np.testing.assert_array_equal(
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=bool),
            is_unlabeled(
                self.y9, missing_label=-1, target_type="single-output"
            ),
        )

    def test_label_and_index_helpers_are_task_agnostic(self):
        y = np.array([0, 1.5, None], dtype=object)
        np.testing.assert_array_equal(
            is_unlabeled(y, missing_label=None),
            [False, False, True],
        )
        np.testing.assert_array_equal(
            is_labeled(y, missing_label=None),
            [True, True, False],
        )
        np.testing.assert_array_equal(
            unlabeled_indices(y, missing_label=None), [2]
        )
        np.testing.assert_array_equal(
            labeled_indices(y, missing_label=None), [0, 1]
        )

        invalid_for_both_tasks = np.array([False, 2], dtype=object)
        for helper in (
            is_unlabeled,
            is_labeled,
            unlabeled_indices,
            labeled_indices,
        ):
            with self.subTest(helper=helper.__name__):
                with self.assertRaisesRegex(TypeError, "one label family"):
                    helper(invalid_for_both_tasks)

    def test_mixed_target_error_identifies_types(self):
        with self.assertRaises(TypeError) as caught:
            is_unlabeled(["a", 1])
        self.assertIn("str", str(caught.exception))
        self.assertIn("int", str(caught.exception))

    def test_complex_targets_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "unsupported label dtype"):
            is_unlabeled(np.array([1 + 2j, 3 + 0j, -1 + 0j]), -1)
        y = np.array([1 + 2j, -1], dtype=object)
        with self.assertRaisesRegex(
            TypeError, "unsupported scalar label type"
        ):
            is_unlabeled(y, -1)

    def test_object_targets_reject_non_scalar_entries(self):
        for value in ([0, 1], np.array([0, 1]), np.array(0), {"label": 0}):
            for missing in (np.nan, -1, None):
                with self.subTest(value=repr(value), missing=missing):
                    y = np.empty((2, 1), dtype=object)
                    y[0, 0] = value
                    y[1, 0] = missing
                    with self.assertRaisesRegex(TypeError, "scalar"):
                        is_unlabeled(y, missing)

    def test_numeric_arrays_reject_string_missing_labels(self):
        # Numeric NaN and the string `"nan"` are different values, so the
        # missing label is incompatible instead of matching the missing
        # entries.
        y = np.array([0.0, np.nan, 1.0])
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            is_unlabeled(y, missing_label="nan")
        with self.assertRaisesRegex(ValueError, "contains NaN"):
            is_unlabeled(y.astype(object), missing_label="nan")
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            is_unlabeled(y.tolist(), missing_label="nan")
        np.testing.assert_array_equal(y, [0.0, np.nan, 1.0])

    def test_object_targets_reject_integers_beyond_64_bits(self):
        for missing in (np.nan, -1, None):
            with self.subTest(missing=missing):
                y = np.array([2**200, 2**200 + 1, missing], dtype=object)
                with self.assertRaisesRegex(
                    ValueError, "does not fit a 64-bit integer dtype"
                ):
                    is_unlabeled(y, missing)
                self.assertEqual(y[:2].tolist(), [2**200, 2**200 + 1])

    def test_object_targets_reject_integers_without_a_common_dtype(self):
        y = np.array([-1, 2**64 - 1, np.nan], dtype=object)
        with self.assertRaisesRegex(
            ValueError, "do not fit one 64-bit integer dtype"
        ):
            is_unlabeled(y, np.nan)

    def test_object_targets_accept_numpy_boolean_scalars(self):
        for missing in (np.nan, None):
            with self.subTest(missing=missing):
                y = np.array(
                    [np.bool_(False), np.bool_(True), missing], dtype=object
                )
                np.testing.assert_array_equal(
                    is_unlabeled(y, missing), [False, False, True]
                )

    def test_missing_label_comparison_preserves_large_integer_identity(self):
        y = [2**53, 2**53 + 1]
        targets_variants = [y, np.asarray(y), np.asarray(y, dtype=object)]
        for dtype in (np.int64, np.uint64):
            targets_variants.append(
                np.array([dtype(v) for v in y], dtype=object)
            )
        for targets in targets_variants:
            np.testing.assert_array_equal(
                is_unlabeled(targets, missing_label=float(2**53)),
                [True, False],
            )
        np.testing.assert_array_equal(
            is_unlabeled(
                np.array([np.float64(2**53)], dtype=object),
                missing_label=2**53 + 1,
            ),
            [False],
        )

    def test_is_unlabeled_accepts_numpy_floating_nan_sentinels(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                missing_label = dtype(np.nan)
                y = np.array([0, missing_label], dtype=dtype)

                np.testing.assert_array_equal(
                    is_unlabeled(y, missing_label=missing_label),
                    [False, True],
                )

    def test_is_unlabeled_rejects_extended_precision_sentinels(self):
        missing_label = np.longdouble(np.nan)
        y = np.array([0, missing_label], dtype=np.longdouble)
        with self.assertRaisesRegex(TypeError, "must be a number"):
            is_unlabeled(y, missing_label=missing_label)
        with self.assertRaisesRegex(TypeError, "unsupported label dtype"):
            is_unlabeled(y, missing_label=np.nan)

    def test_is_unlabeled_rejects_complex_nan_sentinels(self):
        for missing_label in (
            complex(float("nan"), 0),
            np.complex64(np.nan),
            np.complex128(np.nan),
        ):
            with self.subTest(dtype=type(missing_label)):
                y = np.array([0, missing_label])

                with self.assertRaisesRegex(TypeError, "must be a number"):
                    is_unlabeled(y, missing_label=missing_label)

    def test_is_unlabeled_rejects_nan_under_a_none_missing_label(self):
        # Only the configured missing label denotes missingness.
        for targets in (
            [0.0, np.nan, 1.0],
            np.array([0.0, np.nan, 1.0]),
            np.array([0.0, np.nan, 1.0], dtype=object),
        ):
            with self.subTest(targets=repr(targets)):
                with self.assertRaisesRegex(ValueError, "contains NaN"):
                    is_unlabeled(targets, missing_label=None)

    def test_is_unlabeled_rejects_infinite_labels_and_missing_labels(self):
        for targets in (
            [0.0, np.inf],
            np.array([0.0, -np.inf]),
            np.array([0.0, np.inf], dtype=object),
        ):
            with self.subTest(targets=repr(targets)):
                with self.assertRaisesRegex(
                    ValueError, "contains an infinite value"
                ):
                    is_unlabeled(targets, missing_label=np.nan)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            is_unlabeled([0.0, 1.0], missing_label=np.inf)

    def test_arrays_mixing_numbers_and_strings_are_rejected(self):
        for targets in (
            ["paris", -1],
            np.array(["paris", -1], dtype=object),
        ):
            with self.subTest(targets=repr(targets)):
                with self.assertRaisesRegex(TypeError, "one label family"):
                    is_unlabeled(targets, missing_label=np.nan)

    def test_observed_numerical_labels_are_floating_point_values(self):
        for missing in (np.nan, None, -999):
            with self.subTest(missing=missing):
                y = [0, 1.5, missing]
                is_lbld, observed = _observed_numerical_labels(y, missing)
                np.testing.assert_array_equal(is_lbld, [True, True, False])
                self.assertEqual(observed.dtype, np.dtype(float))
                np.testing.assert_array_equal(observed, [0.0, 1.5])

    def test_observed_numerical_labels_reject_string_values(self):
        for y in ([True, False], ["0.5", "1.5"]):
            with self.subTest(y=y):
                with self.assertRaisesRegex(TypeError, "numerical labels"):
                    _observed_numerical_labels(y, np.nan)

    def test_is_labeled(self):
        np.testing.assert_array_equal(
            ~np.array([1, 0, 0, 0, 1], dtype=bool), is_labeled(self.y1)
        )
        np.testing.assert_array_equal(
            ~np.array([1, 0, 0, 0, 1], dtype=bool),
            is_labeled(self.y3, missing_label=None),
        )
        np.testing.assert_array_equal(
            ~np.array([1, 0, 0, 0, 1], dtype=bool),
            is_labeled(self.y4, missing_label=None),
        )
        np.testing.assert_array_equal(
            ~np.array([0, 0, 0, 0, 0], dtype=bool),
            is_labeled(self.y5, missing_label=None),
        )
        np.testing.assert_array_equal(
            ~np.array([0, 0, 0, 0, 0], dtype=bool),
            is_labeled(self.y5, missing_label=np.nan),
        )
        np.testing.assert_array_equal(
            ~np.array([0, 1, 0, 0, 0], dtype=bool),
            is_labeled(self.y5, missing_label=-1),
        )
        np.testing.assert_array_equal(
            ~np.array([0, 0, 0, 0], dtype=bool),
            is_labeled(self.y6, missing_label=None),
        )
        np.testing.assert_array_equal(
            ~np.array([0, 0, 0, 1], dtype=bool),
            is_labeled(self.y6, missing_label="nan"),
        )
        np.testing.assert_array_equal(
            ~np.array([0, 0, 0, 1], dtype=bool),
            is_labeled(self.y8, missing_label=-1, target_type="multi-label"),
        )
        np.testing.assert_array_equal(
            ~np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=bool
            ),
            is_labeled(self.y9, missing_label=-1, target_type="single-output"),
        )

    def test_unlabeled_indices(self):
        unlbld_indices = unlabeled_indices(self.y3, missing_label=None)
        true_unlbld_indices = [0, 4]
        np.testing.assert_array_equal(unlbld_indices, true_unlbld_indices)
        y = np.array([self.y3]).T
        unlbld_indices = unlabeled_indices(y, missing_label=None)
        true_unlbld_indices = [[0, 0], [4, 0]]
        np.testing.assert_array_equal(unlbld_indices, true_unlbld_indices)

    def test_labeled_indices(self):
        lbld_indices = labeled_indices(self.y3, missing_label=None)
        true_lbld_indices = [1, 2, 3]
        np.testing.assert_array_equal(lbld_indices, true_lbld_indices)
        y = np.array([self.y3]).T
        lbld_indices = labeled_indices(y, missing_label=None)
        true_lbld_indices = [[1, 0], [2, 0], [3, 0]]
        np.testing.assert_array_equal(lbld_indices, true_lbld_indices)

    def test_check_missing_label(self):
        self.assertRaises(TypeError, check_missing_label, missing_label=[2])
        self.assertRaises(TypeError, check_missing_label, missing_label=self)
        self.assertRaises(
            TypeError,
            check_missing_label,
            missing_label=np.nan,
            target_type=str,
        )
        self.assertRaises(
            TypeError, check_missing_label, missing_label=2, target_type=str
        )
        self.assertRaises(
            TypeError, check_missing_label, missing_label="2", target_type=int
        )

    def test_check_equal_missing_label(self):
        self.assertRaises(
            ValueError,
            check_equal_missing_label,
            missing_label1=np.nan,
            missing_label2=None,
        )
