import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np

from skactiveml.utils._label_dtype import (
    _as_label_array,
    _check_missing_label_for_kind,
    _check_missing_label_value,
    _common_label_dtype,
    _holds_missing_label,
    _label_kind,
    _matches_missing_label,
    _missing_mask_and_kind,
)


class TestAsLabelArray(unittest.TestCase):
    def test_preserves_ordinary_sequence_dtypes(self):
        cases = [
            ([0, 1], np.dtype(np.int64)),
            ([0, np.nan, 1], np.dtype(np.float64)),
            ((-0.5, 1.5), np.dtype(np.float64)),
            ([False, 2], np.dtype(np.int64)),
            (["a", "long"], np.dtype("U4")),
            ([b"a", b"long"], np.dtype("S4")),
            ([], np.dtype(np.float64)),
        ]
        for values, dtype in cases:
            with self.subTest(values=values):
                result = _as_label_array(values)
                self.assertEqual(result.dtype, dtype)
                np.testing.assert_array_equal(result, values)

    def test_preserves_structured_targets_and_nonfinite_values(self):
        values = [[0, np.nan], [np.inf, -np.inf]]
        result = _as_label_array(values)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.dtype(float))
        np.testing.assert_array_equal(result, values)
        complex_values = [1 + 2j, 3 + 0j]
        np.testing.assert_array_equal(
            _as_label_array(complex_values), complex_values
        )

    def test_does_not_stringify_mixed_values(self):
        for values in (
            [1, "missing"],
            ["a", None],
            [b"a", "b"],
            [[0, "a"], [1, "b"]],
        ):
            with self.subTest(values=values):
                result = _as_label_array(values)
                self.assertEqual(result.dtype, np.dtype(object))
                self.assertEqual(result.tolist(), values)
        self.assertIs(_as_label_array(["a", None])[1], None)
        self.assertTrue(np.isnan(_as_label_array(["cat", np.nan])[1]))

    def test_reuses_existing_arrays_including_empty_shapes(self):
        for values in (
            np.array([0.0, np.nan]),
            np.array([0, "1"]),
            np.array([0, 1.5], dtype=object),
            np.empty((0, 2)),
        ):
            with self.subTest(dtype=values.dtype, shape=values.shape):
                self.assertIs(_as_label_array(values), values)


class TestCommonLabelDtype(unittest.TestCase):
    def test_omitted_missing_label_differs_from_explicit_none(self):
        classes = [np.array([0, 1], dtype=np.int32)]
        self.assertEqual(_common_label_dtype(classes), np.dtype(np.int32))
        self.assertEqual(_common_label_dtype(classes, None), np.dtype(object))
        self.assertEqual(_common_label_dtype(classes, np.nan), np.dtype(float))

    def test_numeric_vocabularies_share_numpys_common_dtype(self):
        classes = [np.array([0, 1]), np.array([0.0, 1.0])]
        self.assertEqual(_common_label_dtype(classes), np.dtype(float))
        self.assertEqual(
            _common_label_dtype([np.array([False, True])], -1),
            np.dtype(np.int64),
        )

    def test_strings_beside_other_values_use_object_storage(self):
        self.assertEqual(
            _common_label_dtype([np.array([b"a"]), np.array(["b"])]),
            np.dtype(object),
        )
        self.assertEqual(
            _common_label_dtype([np.array(["a"]), np.array([0, 1])]),
            np.dtype(object),
        )
        self.assertEqual(
            _common_label_dtype([np.array(["a"])], "missing"),
            np.dtype("U7"),
        )
        self.assertEqual(
            _common_label_dtype([np.array(["a"])], np.nan),
            np.dtype(object),
        )


class TestLabelKind(unittest.TestCase):
    def test_dtype_decides_the_kind_of_an_ordinary_array(self):
        cases = [
            (np.array([False, True]), "number"),
            (np.array([0, 1], dtype=np.int8), "number"),
            (np.array([0, 1], dtype=np.uint64), "number"),
            (np.array([0.0, 1.5], dtype=np.float16), "number"),
            (np.array([0.0, 1.5], dtype=np.longdouble), "number"),
            (np.array(["cat", "dog"]), "str"),
            (np.array([[0, 1], [2, 3]]), "number"),
        ]
        for values, kind in cases:
            with self.subTest(dtype=values.dtype):
                self.assertEqual(_label_kind(values, name="y"), kind)

    def test_object_arrays_are_checked_by_the_types_of_their_values(self):
        cases = [
            ([np.bool_(True), False], "number"),
            ([np.int8(0), 1, np.uint64(2), 2**70], "number"),
            ([False, 2], "number"),
            ([0, 1.5, np.float32(0.5)], "number"),
            ([np.str_("cat"), "dog"], "str"),
        ]
        for values, kind in cases:
            with self.subTest(values=repr(values)):
                y = np.array(values, dtype=object)
                self.assertEqual(_label_kind(y, name="y"), kind)

    def test_empty_values_provide_no_evidence(self):
        for values in ([], np.array([]), np.empty((0, 2), dtype=object)):
            with self.subTest(values=repr(values)):
                self.assertIsNone(_label_kind(values, name="y"))

    def test_rejects_dtypes_outside_the_contract(self):
        cases = [
            np.array([1 + 2j]),
            np.array([b"cat"]),
            np.array(["2026-09-08"], dtype="M8[D]"),
            np.array([1], dtype="m8[s]"),
        ]
        for values in cases:
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(TypeError, "unsupported type"):
                    _label_kind(values, name="y")

    def test_rejects_scalar_types_outside_the_contract(self):
        cases = [
            1 + 2j,
            np.complex64(1),
            b"cat",
            Decimal("1.5"),
            Fraction(1, 2),
            np.timedelta64(3, "ns"),
            np.datetime64("2026-09-08"),
            np.array([0, 1]),
            [0, 1],
            {"label": 0},
        ]
        for value in cases:
            with self.subTest(value=repr(value)):
                y = np.empty(1, dtype=object)
                y[0] = value
                with self.assertRaisesRegex(TypeError, "unsupported type"):
                    _label_kind(y, name="y")

    def test_rejects_none_naming_the_missing_label(self):
        y = np.array([0, None], dtype=object)
        with self.assertRaisesRegex(TypeError, "only valid as"):
            _label_kind(y, name="y")

    def test_rejects_nonfinite_values(self):
        for values in (
            np.array([0.0, np.nan]),
            np.array([0, np.nan], dtype=object),
        ):
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(ValueError, "contains NaN"):
                    _label_kind(values, name="y")
        for values in (
            np.array([0.0, np.inf]),
            np.array([0, -np.inf], dtype=object),
        ):
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(
                    ValueError, "contains an infinite value"
                ):
                    _label_kind(values, name="y")

    def test_rejects_strings_mixed_with_numbers(self):
        cases = [
            np.array(["cat", 1], dtype=object),
            np.array([np.bool_(True), "cat"], dtype=object),
            np.array([[1.5], ["cat"]], dtype=object),
        ]
        for values in cases:
            with self.subTest(values=repr(values)):
                with self.assertRaisesRegex(
                    TypeError, "`classes` mixes strings and numbers"
                ):
                    _label_kind(values, name="classes")


class TestMissingLabelChecks(unittest.TestCase):
    def test_accepts_supported_missing_label_values(self):
        for missing_label in (
            None,
            "?",
            np.str_("?"),
            np.nan,
            np.float16(np.nan),
            np.longdouble(np.nan),
            -2.5,
            -1,
            np.int8(-1),
            np.uint64(2**63),
            2**64,
        ):
            with self.subTest(missing_label=repr(missing_label)):
                _check_missing_label_value(missing_label)

    def test_rejects_unsupported_missing_label_values(self):
        for missing_label in (
            [2],
            {"missing": 1},
            1 + 2j,
            np.complex128(np.nan),
            b"?",
            True,
            np.bool_(False),
            np.timedelta64(1, "s"),
        ):
            with self.subTest(missing_label=repr(missing_label)):
                with self.assertRaisesRegex(
                    TypeError, "must be a real number"
                ):
                    _check_missing_label_value(missing_label)

    def test_rejects_infinite_missing_labels(self):
        for missing_label in (np.inf, -np.inf, np.float32(np.inf)):
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    _check_missing_label_value(missing_label)

    def test_missing_label_compatibility_follows_the_contract(self):
        numeric_missing_labels = (np.nan, -2.5, -1)
        _check_missing_label_for_kind(None, "number", name="y")
        for missing_label in numeric_missing_labels:
            _check_missing_label_for_kind(missing_label, "number", name="y")
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            _check_missing_label_for_kind("?", "number", name="y")
        _check_missing_label_for_kind(None, "str", name="y")
        _check_missing_label_for_kind("?", "str", name="y")
        for missing_label in numeric_missing_labels:
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(
                    TypeError, "is not compatible with"
                ):
                    _check_missing_label_for_kind(
                        missing_label, "str", name="y"
                    )

    def test_unknown_kind_accepts_every_missing_label(self):
        for missing_label in (None, "?", np.nan, -1):
            with self.subTest(missing_label=repr(missing_label)):
                _check_missing_label_for_kind(missing_label, None, name="y")

    def test_missing_label_error_names_the_labels(self):
        with self.assertRaises(TypeError) as caught:
            _check_missing_label_for_kind("?", "number", name="classes")
        self.assertIn("`classes`", str(caught.exception))
        with self.assertRaises(TypeError) as caught:
            _check_missing_label_for_kind("?", "number")
        self.assertIn("target object", str(caught.exception))

    def test_numeric_nan_never_equals_the_string_missing_label(self):
        self.assertFalse(_matches_missing_label(np.nan, "nan"))
        self.assertFalse(_matches_missing_label("nan", np.nan))
        self.assertTrue(_matches_missing_label(np.nan, np.float32(np.nan)))
        self.assertTrue(_matches_missing_label(None, None))
        self.assertFalse(_matches_missing_label(None, np.nan))
        self.assertTrue(_matches_missing_label(-1, -1.0))

    def test_only_a_dtype_holding_the_missing_label_can_store_it(self):
        cases = [
            (np.array([0, 1]), np.nan, False),
            (np.array([0.0, 1.0]), np.nan, True),
            (np.array([0.0, 1.0]), None, False),
            (np.array([0.0, 1.0]), -1, True),
            (np.array([0.0, 1.0], dtype=np.float32), np.nan, True),
            (np.array([0, 1], dtype=np.int8), -1, True),
            (np.array([0, 1], dtype=np.int8), 127, True),
            (np.array([0, 1], dtype=np.int8), np.int8(127), True),
            (np.array([0, 1], dtype=np.uint8), -1, False),
            (np.array([0.0, 1.0], dtype=np.float32), 0.1, False),
            (np.array([0.0, 1.0], dtype=np.float32), np.float64(0.1), False),
            (np.array([False, True]), -1, False),
            (np.array(["cat", "dog"]), "?", True),
            (np.array(["cat", "dog"]), "unknown", False),
            (np.array([0, 1], dtype=object), None, True),
        ]
        for values, missing_label, expected in cases:
            with self.subTest(
                dtype=values.dtype, missing_label=repr(missing_label)
            ):
                self.assertEqual(
                    _holds_missing_label(values, missing_label), expected
                )


class TestMissingMaskAndKind(unittest.TestCase):
    def test_missing_label_is_checked_against_the_dtype_of_an_array(self):
        y = np.array([0.0, np.nan, 1.0])
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            _missing_mask_and_kind(y, "nan", name="y")

    def test_observed_values_of_an_object_array_decide_the_kind(self):
        y = np.array(["cat", "?", "dog"], dtype=object)
        is_missing, kind = _missing_mask_and_kind(y, "?", name="y")
        np.testing.assert_array_equal(is_missing, [False, True, False])
        self.assertEqual(kind, "str")
        y = np.array(["cat", -1, "dog"], dtype=object)
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            _missing_mask_and_kind(y, -1, name="y")

    def test_nan_among_observed_values_is_rejected(self):
        for y in (
            np.array([0.0, np.nan, 1.0]),
            np.array([0.0, np.nan, 1.0], dtype=object),
        ):
            with self.subTest(dtype=y.dtype):
                with self.assertRaisesRegex(ValueError, "contains NaN"):
                    _missing_mask_and_kind(y, None, name="y")

    def test_missing_label_of_another_kind_marks_nothing(self):
        y = np.array(["cat", "dog"])
        is_missing, kind = _missing_mask_and_kind(y, None, name="y")
        np.testing.assert_array_equal(is_missing, [False, False])
        self.assertEqual(kind, "str")
        y = np.array([0, 1])
        is_missing, kind = _missing_mask_and_kind(y, np.nan, name="y")
        np.testing.assert_array_equal(is_missing, [False, False])
        self.assertEqual(kind, "number")

    def test_entirely_missing_labels_provide_no_kind_evidence(self):
        cases = [
            (np.array([np.nan, np.nan]), np.nan),
            (np.array([np.nan, np.nan], dtype=np.longdouble), np.nan),
            (np.array([-1, -1]), -1),
            (np.array(["?", "?"]), "?"),
            (np.array([None, None], dtype=object), None),
            (np.array([np.nan, np.float32(np.nan)], dtype=object), np.nan),
        ]
        for y, missing_label in cases:
            with self.subTest(dtype=y.dtype, missing_label=missing_label):
                is_missing, kind = _missing_mask_and_kind(
                    y, missing_label, name="y"
                )
                self.assertTrue(is_missing.all())
                self.assertIsNone(kind)

    def test_numeric_missing_label_is_not_rounded_to_the_label_dtype(self):
        for y, missing_label in (
            (np.array([16777216.0, 1.0], dtype=np.float32), 16777217),
            (np.array([1.0, 2.0], dtype=np.float32), 1.00000001),
        ):
            with self.subTest(missing_label=missing_label):
                is_missing, _ = _missing_mask_and_kind(
                    y, missing_label, name="y"
                )
                np.testing.assert_array_equal(is_missing, [False, False])

    def test_numbers_of_every_type_may_be_mixed(self):
        cases = [
            np.array([0, 1.5, None], dtype=object),
            np.array([np.bool_(True), 1, None], dtype=object),
            np.array([True, np.float32(1.5), None], dtype=object),
        ]
        for y in cases:
            with self.subTest(y=repr(y)):
                is_missing, kind = _missing_mask_and_kind(y, None, name="y")
                np.testing.assert_array_equal(is_missing, [False, False, True])
                self.assertEqual(kind, "number")

    def test_numerical_labels_reject_boolean_and_string_values(self):
        for y in (
            np.array([True, False]),
            np.array(["0.5"]),
            np.array([0.5, True], dtype=object),
        ):
            with self.subTest(y=repr(y)):
                with self.assertRaisesRegex(TypeError, "numerical labels"):
                    _missing_mask_and_kind(y, np.nan, name="y", numerical=True)
        y = np.array([0, 1.5, np.nan], dtype=object)
        is_missing, kind = _missing_mask_and_kind(
            y, np.nan, name="y", numerical=True
        )
        np.testing.assert_array_equal(is_missing, [False, False, True])
        self.assertEqual(kind, "number")

    def test_incompatible_missing_label_names_the_observed_kind(self):
        cases = [
            (np.array([1, 2]), "?", "numeric"),
            (np.array([True, False]), "?", "numeric"),
            (np.array([1, 2.5], dtype=object), "?", "numeric"),
            (np.array(["a", "b"]), -1, "string"),
            (np.array(["a", "b"], dtype=object), np.nan, "string"),
        ]
        for y, missing_label, description in cases:
            with self.subTest(y=repr(y)):
                with self.assertRaisesRegex(
                    TypeError, f"the {description} labels in `y`"
                ):
                    _missing_mask_and_kind(y, missing_label, name="y")
