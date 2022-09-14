import unittest

import numpy as np

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

    def test_is_unlabeled(self):
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y1, missing_label="2"
        )
        self.assertRaises(ValueError, is_unlabeled, [[]], missing_label="2")
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y2, missing_label=np.nan
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y2, missing_label="2"
        )
        self.assertRaises(
            TypeError, is_unlabeled, y=self.y2, missing_label=None
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
