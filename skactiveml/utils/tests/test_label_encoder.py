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

    def test_mixed_integer_vocabularies_are_rejected(self):
        # One array holds every output of a sample, so a signed vocabulary
        # beside an unsigned one has no common dtype to store them in.
        signed = np.array([2**53, 2**53 + 1], dtype=np.int64)
        unsigned = np.array([2**63, 2**63 + 1], dtype=np.uint64)
        encoder = ExtLabelEncoder(
            classes=[signed, unsigned], target_type="multi-label"
        )
        with self.assertRaisesRegex(
            ValueError, "one dtype across all label outputs"
        ):
            encoder.fit(np.empty((0, 2)))

    def test_targets_beyond_64_bits_are_rejected(self):
        labels = [2**200, 2**200 + 1]
        for declared in (False, True):
            with self.subTest(declared=declared):
                encoder = ExtLabelEncoder(classes=labels if declared else None)
                with self.assertRaisesRegex(
                    ValueError, "does not fit a 64-bit integer dtype"
                ):
                    encoder.fit([*labels, np.nan])

    def test_raw_list_targets_preserve_integer_identity(self):
        for labels in (
            [2**53, 2**53 + 1],
            [2**63, 2**63 + 1],
        ):
            for missing in (np.nan, None, -2.5, -1):
                for declared in (False, True):
                    with self.subTest(
                        labels=repr(labels), missing=missing, declared=declared
                    ):
                        y = [*labels, missing]
                        encoder = ExtLabelEncoder(
                            classes=labels if declared else None,
                            missing_label=missing,
                        )
                        codes = encoder.fit_transform(y)
                        np.testing.assert_array_equal(codes, [0, 1, -1])
                        self.assertEqual(
                            encoder.inverse_transform(codes)[:2].tolist(),
                            labels,
                        )
                        self.assertEqual(y[:2], labels)

    def test_mixed_numpy_integer_scalar_classes_preserve_identity(self):
        labels = [np.int64(2**53), np.uint64(2**53 + 1)]
        encoder = ExtLabelEncoder(classes=labels).fit([])

        self.assertEqual(encoder.classes_.dtype, np.dtype(np.int64))
        self.assertEqual(encoder.classes_.tolist(), [2**53, 2**53 + 1])
        np.testing.assert_array_equal(encoder.transform(labels), [0, 1])

    def test_integer_label_round_trip_is_exact(self):
        vocabularies = [
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            np.array([-(2**63), 2**63 - 1], dtype=np.int64),
            np.array([2**63, 2**63 + 1], dtype=np.uint64),
            np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
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
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
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
                        # Fit on the lossless representation, too.
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

    def test_inverse_transform_preserves_lossless_dtypes(self):
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
            (np.array([2**53, 2**53 + 1], dtype=np.int64), np.nan),
            (np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64), np.nan),
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
            ExtLabelEncoder(classes=classes, target_type="multi-output").fit,
            [["a", "c"], ["b", "d"]],
        )
