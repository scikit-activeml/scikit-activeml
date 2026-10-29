import unittest

import numpy as np
from sklearn.metrics import confusion_matrix

from skactiveml.utils import ext_confusion_matrix


class TestMultiAnnot(unittest.TestCase):
    def test_ext_confusion_matrix_accepts_containers_and_missing_labels(self):
        expected = np.array(
            [[[0, 1], [0, 0]], [[1, 0], [1, 0]], [[0, 0], [0, 0]]]
        )
        for labels in ([3, 7], [-7, -3], [0.5, 1.5]):
            for missing_label in (np.nan, None, -1):
                predictions = [
                    [labels[1], labels[0], missing_label],
                    [missing_label, labels[0], missing_label],
                ]
                for container in (list, tuple, "object"):
                    if container == "object":
                        y_true = np.array(labels, dtype=object)
                        y_pred = np.array(predictions, dtype=object)
                    else:
                        y_true = container(labels)
                        y_pred = container(predictions)
                    for classes in (None, labels, labels[::-1]):
                        with self.subTest(
                            labels=labels,
                            missing_label=missing_label,
                            container=container,
                            classes=classes,
                        ):
                            actual = ext_confusion_matrix(
                                y_true,
                                y_pred,
                                classes=classes,
                                missing_label=missing_label,
                            )
                            np.testing.assert_array_equal(
                                actual,
                                (
                                    expected[:, ::-1, ::-1]
                                    if classes == labels[::-1]
                                    else expected
                                ),
                            )

    def test_ext_confusion_matrix_accepts_mixed_numeric_array_dtypes(self):
        cases = [
            (
                np.array([7, 0], dtype=np.int64),
                np.array([8, 0], dtype=np.uint64),
                [0, 7, 8],
                [[[1, 0, 0], [0, 0, 1], [0, 0, 0]]],
            ),
            (
                np.array([3, 0], dtype=np.int64),
                np.array([0.0, 0.0]),
                [0, 3],
                [[[1, 0], [1, 0]]],
            ),
            (
                np.array([3, 7], dtype=np.int64),
                np.array([np.nan, np.nan]),
                [3, 7],
                [[[0, 0], [0, 0]]],
            ),
        ]
        for y_true, y_pred, labels, expected in cases:
            for classes in (None, labels):
                with self.subTest(
                    y_true=y_true, y_pred=y_pred, classes=classes
                ):
                    actual = ext_confusion_matrix(
                        y_true, y_pred, classes=classes
                    )
                    np.testing.assert_array_equal(actual, expected)

    def test_ext_confusion_matrix_preserves_requested_class_order(self):
        actual = ext_confusion_matrix([0, 0, 1], [0, 1, 1], classes=[1, 0])

        np.testing.assert_array_equal(
            actual,
            [[[1, 0], [1, 1]]],
        )

    def test_ext_confusion_matrix_selects_requested_class_subset(self):
        actual = ext_confusion_matrix([0, 0, 1], [0, 1, 1], classes=[0])

        np.testing.assert_array_equal(actual, [[[1]]])

    def test_ext_confusion_matrix_keeps_unobserved_fractional_classes(self):
        actual = ext_confusion_matrix(
            [0, 1, 1], [[0, 1], [1, 1], [1, 0]], classes=[0.5, 1]
        )

        np.testing.assert_array_equal(
            actual, [[[0, 0], [0, 2]], [[0, 0], [0, 1]]]
        )

    def test_ext_confusion_matrix_order_with_missing_annotator_labels(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array(
            [
                [0, 1],
                [0, 0],
                [1, np.nan],
                [0, 0],
                [0, 1],
                [0, 0],
                [np.nan, 0],
            ]
        )
        classes = [1, 0]

        for normalize in (None, "true", "pred", "all"):
            with self.subTest(normalize=normalize):
                actual = ext_confusion_matrix(
                    y_true,
                    y_pred,
                    classes=classes,
                    normalize=normalize,
                )
                expected = []
                for predictions in y_pred.T:
                    observed = ~np.isnan(predictions)
                    expected.append(
                        confusion_matrix(
                            y_true[observed],
                            predictions[observed],
                            labels=classes,
                            normalize=normalize,
                        )
                    )

                np.testing.assert_array_equal(actual, expected)

    def test_ext_confusion_matrix_keeps_unobserved_requested_classes(self):
        actual = ext_confusion_matrix([0, 0, 1], [0, 1, 1], classes=[2, 1, 0])

        np.testing.assert_array_equal(
            actual,
            [[[0, 0, 0], [0, 1, 0], [0, 1, 1]]],
        )

    def test_ext_confusion_matrix(self):
        for y_true, y_pred in (
            ([0, 1], ["0", "1"]),
            (["0", "1"], [0, 1]),
            ([0, 1], [[0, "0"], [1, "1"]]),
            ([0, "1"], [0, 1]),
        ):
            with self.subTest(y_true=y_true, y_pred=y_pred):
                self.assertRaises(
                    TypeError,
                    ext_confusion_matrix,
                    y_true=y_true,
                    y_pred=y_pred,
                    missing_label=None,
                )
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

        actual = ext_confusion_matrix(
            ["a", "a", "b"],
            [["a", "b", None], ["b", None, None], ["b", "a", None]],
            classes=["a", "b"],
            missing_label=None,
        )
        np.testing.assert_array_equal(
            actual, [[[1, 1], [0, 1]], [[0, 1], [1, 0]], [[0, 0], [0, 0]]]
        )
