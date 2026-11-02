import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from skactiveml.utils import ExtLabelEncoder


class TestLabelEncoder(unittest.TestCase):
    def setUp(self):
        self.y1 = [np.nan, 2, 5, 10, np.nan]
        self.y2 = [np.nan, "2", "5", "10", np.nan]
        self.y3 = [None, 2, 5, 10, None]
        self.y4 = [None, "2", "5", "10", None]
        self.y5 = [8, -1, 1, 5, 2]
        self.y6 = ["paris", "france", "tokyo", "nan"]
        self.y7 = ["paris", "france", "tokyo", -1]

    def test_inferred_classes_follow_numpy_conversion(self):
        for y, dtype in (
            (np.array([0, 1, None], dtype=object), np.dtype(np.int64)),
            (np.array([False, True, None], dtype=object), np.dtype(bool)),
            (np.array([0, 1.5, None], dtype=object), np.dtype(float)),
            (np.array(["a", "b", None], dtype=object), np.dtype("U1")),
        ):
            with self.subTest(y=y.tolist()):
                encoder = ExtLabelEncoder(missing_label=None).fit(y)
                self.assertEqual(encoder.classes_.dtype, dtype)

    def test_transform_rejects_labels_outside_the_classes(self):
        encoder = ExtLabelEncoder(classes=[0, 1]).fit([0, 1])
        with self.assertRaisesRegex(ValueError, "outside `classes_`"):
            encoder.transform([0.5, 1.0])
        encoder = ExtLabelEncoder(
            classes=[[0, 1], [0, 1]], target_type="multi-label"
        ).fit([[0, 1]])
        with self.assertRaisesRegex(ValueError, "outside `classes_`"):
            encoder.transform([[0, 0.5]])

    def test_multilabel_fit_rejects_labels_of_another_kind(self):
        encoder = ExtLabelEncoder(
            classes=[["a", "b"], ["c", "d"]],
            missing_label=None,
            target_type="multi-label",
        )
        with self.assertRaisesRegex(TypeError, r"classes\[0\]"):
            encoder.fit([[0, 1], [1, 0]])

    def test_integer_label_round_trip_is_exact(self):
        vocabularies = [
            np.array([-3, 7], dtype=np.int64),
            np.array([-3, 7], dtype=np.int8),
            np.array([3, 7], dtype=np.uint64),
        ]
        for vocabulary in vocabularies:
            for declared in (False, True):
                for missing in (np.nan, None, -2.5, -1):
                    with self.subTest(
                        vocabulary=repr(vocabulary),
                        declared=declared,
                        missing=missing,
                    ):
                        encoder = ExtLabelEncoder(
                            classes=vocabulary if declared else None,
                            missing_label=missing,
                        ).fit(vocabulary)
                        decoded = encoder.inverse_transform([0, 1])
                        self.assertEqual(decoded.tolist(), vocabulary.tolist())
                        decoded = encoder.inverse_transform([0, 1, -1])
                        self.assertEqual(
                            decoded[:2].tolist(), vocabulary.tolist()
                        )
                        if missing is None:
                            self.assertIs(decoded[-1], None)
                        elif np.isnan(missing):
                            self.assertTrue(np.isnan(decoded[-1]))
                        else:
                            self.assertEqual(decoded[-1], missing)
                        np.testing.assert_array_equal(
                            encoder.transform(decoded), [0, 1, -1]
                        )
                        np.testing.assert_array_equal(
                            encoder.fit_transform(decoded), [0, 1, -1]
                        )

    def test_structured_integer_label_round_trip_is_exact(self):
        for vocabulary in (
            np.array([3, 7], dtype=np.int64),
            np.array([3, 7], dtype=np.uint64),
        ):
            for target_type in ("single-output", "multi-label"):
                for missing in (np.nan, None, -2.5, -1):
                    with self.subTest(
                        vocabulary=repr(vocabulary),
                        target_type=target_type,
                        missing=missing,
                    ):
                        multi_label = target_type == "multi-label"
                        classes = (
                            [vocabulary, vocabulary]
                            if multi_label
                            else vocabulary
                        )
                        encoder = ExtLabelEncoder(
                            classes=classes,
                            missing_label=missing,
                            target_type=target_type,
                        ).fit(np.column_stack([vocabulary, vocabulary]))
                        codes = np.array([[0, 1], [1, 0], [-1, -1]])
                        decoded = encoder.inverse_transform(codes)
                        self.assertEqual(
                            decoded[:2].tolist(),
                            [vocabulary.tolist(), vocabulary[::-1].tolist()],
                        )
                        np.testing.assert_array_equal(
                            encoder.transform(decoded), codes
                        )
                        np.testing.assert_array_equal(
                            encoder.fit_transform(decoded), codes
                        )
                        self.assertEqual(
                            encoder.inverse_transform(
                                np.empty((0, 2), dtype=int)
                            ).shape,
                            (0, 2),
                        )
                        if not multi_label:
                            codes[0, 1] = -1
                            np.testing.assert_array_equal(
                                encoder.transform(
                                    encoder.inverse_transform(codes)
                                ),
                                codes,
                            )

    def test_inverse_transform_dtype_holds_the_missing_label(self):
        for classes, missing, dtype in (
            ([0, 1], np.nan, np.dtype(float)),
            ([0, 1], -2.5, np.dtype(float)),
            ([0, 1], -1, np.dtype(int)),
            ([0, 1], None, np.dtype(object)),
            (["a", "b"], "unknown", np.dtype("U7")),
            (["a", "b"], None, np.dtype(object)),
        ):
            with self.subTest(classes=classes, missing=missing):
                encoder = ExtLabelEncoder(
                    classes=classes, missing_label=missing
                ).fit(classes)
                for codes in ([0, 1], [0, 1, -1], []):
                    decoded = encoder.inverse_transform(codes)
                    self.assertEqual(decoded.dtype, dtype)
                    self.assertEqual(decoded.shape, (len(codes),))
                    np.testing.assert_array_equal(
                        encoder.transform(decoded), codes
                    )

    def test_inverse_transform_prefer_class_dtype(self):
        cases = [
            (np.array([-1, 20]), np.nan),
            (np.array([10, 20]), None),
            (np.array([10, 20]), -2.5),
            (np.array([10, 20]), -1),
            (np.array([10, 20], dtype=np.uint64), np.nan),
            (np.array(["a", "b"]), "unknown"),
            (np.array(["a", "b"]), None),
        ]
        for classes, missing in cases:
            for layout in ("vector", "annotator-matrix", "multi-label"):
                with self.subTest(
                    classes=repr(classes), missing=missing, layout=layout
                ):
                    multi_label = layout == "multi-label"
                    observed = np.array([0, 1])
                    if layout != "vector":
                        observed = np.array([[0, 1], [1, 0]])
                    missing_codes = np.full_like(observed[:1], -1)
                    encoder = ExtLabelEncoder(
                        classes=[classes, classes] if multi_label else classes,
                        missing_label=missing,
                        target_type=(
                            "multi-label" if multi_label else "single-output"
                        ),
                    ).fit(
                        np.column_stack([classes, classes])
                        if multi_label
                        else classes
                    )
                    for codes in (
                        observed,
                        observed[:0],
                        missing_codes,
                        np.concatenate([observed, missing_codes]),
                    ):
                        original_codes = codes.copy()
                        default = encoder.inverse_transform(codes)
                        explicit_default = encoder.inverse_transform(
                            codes, prefer_class_dtype=False
                        )
                        preferred = encoder.inverse_transform(
                            codes, prefer_class_dtype=True
                        )
                        self.assertEqual(explicit_default.dtype, default.dtype)
                        self.assertEqual(preferred.shape, codes.shape)
                        self.assertEqual(
                            preferred.dtype,
                            (
                                default.dtype
                                if np.any(codes == -1)
                                else classes.dtype
                            ),
                        )
                        for decoded in (explicit_default, preferred):
                            np.testing.assert_array_equal(
                                encoder.transform(decoded), codes
                            )
                            for code, value in zip(codes.flat, decoded.flat):
                                if code == -1:
                                    if missing is None:
                                        self.assertIs(value, None)
                                    elif isinstance(
                                        missing, float
                                    ) and np.isnan(missing):
                                        self.assertTrue(np.isnan(value))
                                    else:
                                        self.assertEqual(value, missing)
                                else:
                                    self.assertEqual(
                                        np.asarray(value).item(),
                                        classes[code].item(),
                                    )
                        np.testing.assert_array_equal(codes, original_codes)

    def test_inverse_transform_prefer_class_dtype_validation(self):
        encoder = ExtLabelEncoder(classes=[10, 20]).fit([])
        for invalid in (None, 0, 1, "yes", []):
            with self.subTest(value=repr(invalid)):
                with self.assertRaisesRegex(TypeError, "prefer_class_dtype"):
                    encoder.inverse_transform(
                        [0, 1], prefer_class_dtype=invalid
                    )
        with self.assertRaises(TypeError):
            encoder.inverse_transform([0, 1], True)
        self.assertEqual(
            encoder.inverse_transform(
                [0, 1], prefer_class_dtype=np.bool_(True)
            ).dtype,
            np.dtype(int),
        )
        for prefer_class_dtype in (False, True):
            with self.subTest(prefer_class_dtype=prefer_class_dtype):
                for codes in ([0.9], [2], [-2]):
                    with self.assertRaises(ValueError):
                        encoder.inverse_transform(
                            codes, prefer_class_dtype=prefer_class_dtype
                        )
                multi = ExtLabelEncoder(
                    classes=[[10, 20], [10, 20]], target_type="multi-label"
                ).fit([[10, 10]])
                for codes in ([0, 1], [[0, -1]]):
                    with self.assertRaises(ValueError):
                        multi.inverse_transform(
                            codes, prefer_class_dtype=prefer_class_dtype
                        )

    def test_ExtLabelEncoder(self):
        ext_le = ExtLabelEncoder(classes=[2, "2"])
        self.assertRaises(TypeError, ext_le.fit, self.y1)
        ext_le = ExtLabelEncoder(classes=["1", "2"], missing_label=np.nan)
        self.assertRaises(TypeError, ext_le.fit, self.y1)
        self.assertRaises(
            NotFittedError, ExtLabelEncoder().transform, y=["1", "2"]
        )
        self.assertRaises(
            TypeError, ExtLabelEncoder(missing_label=-1).fit_transform, self.y7
        )

        # missing_label=np.nan
        ext_le = ExtLabelEncoder().fit(self.y1)
        y_enc = ext_le.transform(self.y1)
        np.testing.assert_array_equal([-1, 0, 1, 2, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y1, y_dec)

        # missing_label=None
        ext_le = ExtLabelEncoder(missing_label=None).fit(self.y3)
        y_enc = ext_le.transform(self.y3)
        np.testing.assert_array_equal([-1, 0, 1, 2, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y3, y_dec)
        ext_le = ExtLabelEncoder(missing_label=None).fit(self.y4)
        y_enc = ext_le.transform(self.y4)
        np.testing.assert_array_equal([-1, 1, 2, 0, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y4, y_dec)

        # missing_label=-1
        ext_le = ExtLabelEncoder(missing_label=-1).fit(self.y5)
        y_enc = ext_le.transform(self.y5)
        np.testing.assert_array_equal([3, -1, 0, 2, 1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y5, y_dec)

        # missing_label='nan'
        ext_le = ExtLabelEncoder(missing_label="nan").fit(self.y6)
        y_enc = ext_le.transform(self.y6)
        np.testing.assert_array_equal([1, 0, 2, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y6, y_dec)

        # classes=[0, 2, 5, 10], missing_label=np.nan
        cls = [0, 2, 5, 10]
        np.testing.assert_array_equal(
            [-1, 1, 2, 3, -1],
            ExtLabelEncoder(classes=cls).fit_transform(self.y1),
        )

        # cases with empty y arrays
        np.testing.assert_array_equal(
            [], ExtLabelEncoder(classes=[0, 1]).fit([]).transform([])
        )
        np.testing.assert_array_equal(
            [0, 1], ExtLabelEncoder(classes=[0, 1]).fit([]).transform([0, 1])
        )
        with self.assertRaisesRegex(ValueError, "No class label is observed"):
            ExtLabelEncoder().fit([])
        self.assertRaises(
            ValueError,
            ExtLabelEncoder(classes=[0, 1]).fit([]).transform,
            [1, 3],
        )
        classes = [["a", "b"], ["c", "d"]]
        missing_label = "nan"
        y = [["a", "c"], ["b", "d"], ["nan", "nan"]]
        ext_le = ExtLabelEncoder(
            classes=classes,
            missing_label=missing_label,
            target_type="multi-label",
        )
        y_enc = ext_le.fit_transform(y)
        np.testing.assert_array_equal([[0, 0], [1, 1], [-1, -1]], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(y, y_dec)
        self.assertEqual(ext_le.target_type, "multi-label")

    def test_ExtLabelEncoder_multilabel_shape_validation(self):
        classes = [["a", "b"], ["c", "d"]]
        ext_le = ExtLabelEncoder(
            classes=classes,
            missing_label="nan",
            target_type="multi-label",
        )

        self.assertRaises(ValueError, ext_le.fit, ["a", "c"])

        ext_le.fit([["a", "c"], ["b", "d"]])
        self.assertRaises(ValueError, ext_le.transform, ["a", "c"])
        self.assertRaises(ValueError, ext_le.inverse_transform, [0, 1])

    def test_ExtLabelEncoder_multilabel_rejects_partial_rows(self):
        ext_le = ExtLabelEncoder(
            classes=[["a", "b"], ["c", "d"]],
            missing_label="nan",
            target_type="multi-label",
        )
        partial_y = [["a", "nan"], ["b", "d"]]

        with self.assertRaisesRegex(ValueError, "either only observed labels"):
            ext_le.fit(partial_y)

        ext_le.fit([["a", "c"], ["b", "d"]])
        with self.assertRaisesRegex(ValueError, "either only observed labels"):
            ext_le.transform(partial_y)
        with self.assertRaisesRegex(ValueError, "either only observed labels"):
            ext_le.inverse_transform([[0, -1], [1, 1]])

    def test_ExtLabelEncoder_multilabel_requires_binary_classes(self):
        ext_le = ExtLabelEncoder(
            classes=[["a", "b", "c"], ["d", "e"]],
            missing_label="nan",
            target_type="multi-label",
        )

        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            ext_le.fit([["a", "d"], ["b", "e"]])

    def test_ExtLabelEncoder_inverse_transform_rejects_invalid_codes(self):
        ext_le = ExtLabelEncoder(classes=[10, 20], missing_label=-1).fit([])

        with self.assertRaisesRegex(ValueError, "previously unseen labels"):
            ext_le.inverse_transform([0.9])
        with self.assertRaisesRegex(TypeError, "not compatible"):
            ext_le.inverse_transform(["1"])

    def test_ExtLabelEncoder_target_type_controls_class_structure(self):
        classes = [["a", "b"], ["c", "d"]]

        self.assertRaises(
            ValueError,
            ExtLabelEncoder(classes=classes, target_type="single-output").fit,
            [["a", "c"], ["b", "d"]],
        )
        self.assertRaises(
            ValueError,
            ExtLabelEncoder(classes=classes, target_type="invalid").fit,
            [["a", "c"], ["b", "d"]],
        )
