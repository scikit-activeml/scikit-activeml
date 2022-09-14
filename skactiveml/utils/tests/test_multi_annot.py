import unittest

import numpy as np

from skactiveml.utils import ext_confusion_matrix


class TestMultiAnnot(unittest.TestCase):
    def test_ext_confusion_matrix(self):
        y_true = ["4", "7", None]
        y_pred = ["3", None, "8"]
        self.assertRaises(
            ValueError,
            ext_confusion_matrix,
            y_true=y_true,
            y_pred=y_pred,
            missing_label=None,
        )
        y_true = ["4", "7", "8"]
        self.assertRaises(
            ValueError,
            ext_confusion_matrix,
            y_true=y_true,
            y_pred=y_pred,
            missing_label=None,
            normalize="test",
        )
        conf_matrices = ext_confusion_matrix(
            y_true=y_true, y_pred=y_pred, missing_label=None
        )
        np.testing.assert_array_equal((1, 4, 4), conf_matrices.shape)
        classes = ["3", "4", "7", "8", "9"]
        conf_matrices = ext_confusion_matrix(
            y_true=y_true, y_pred=y_pred, missing_label=None, classes=classes
        )
        np.testing.assert_array_equal((1, 5, 5), conf_matrices.shape)
        y_pred = np.array([["4", "7", "8"], [None, None, None]]).T
        conf_matrices = ext_confusion_matrix(
            y_true=y_true, y_pred=y_pred, missing_label=None, normalize="true"
        )
        np.testing.assert_array_equal(np.eye(3), conf_matrices[0])
        np.testing.assert_array_equal(
            np.ones((3, 3)) * 1 / 3, conf_matrices[1]
        )
        conf_matrices = ext_confusion_matrix(
            y_true=y_true, y_pred=y_pred, missing_label=None, normalize="all"
        )
        self.assertEqual(conf_matrices[0].sum(), 1)
        self.assertEqual(conf_matrices[1].sum(), 1)
        conf_matrices = ext_confusion_matrix(
            y_true=y_true, y_pred=y_pred, missing_label=None, normalize="pred"
        )
        np.testing.assert_array_equal(np.eye(3), conf_matrices[0])
        np.testing.assert_array_equal(
            np.ones((3, 3)) * 1 / 3, conf_matrices[1]
        )
