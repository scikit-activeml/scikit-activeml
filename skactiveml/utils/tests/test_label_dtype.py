import unittest
import unittest.mock
import warnings

import numpy as np

from skactiveml.utils._label_dtype import (
    _LABELS,
    _NUMERICAL_LABELS,
    _TASK_AGNOSTIC_LABELS,
    _as_class_vocabulary_array,
    _as_label_array,
    _check_missing_label_for_family,
    _check_missing_label_value,
    _holds_missing_label,
    _label_family,
    _lossless_decode_dtype,
    _matches_missing_label,
    _missing_mask_and_family,
)


class TestAsLabelArray(unittest.TestCase):
    def test_preserves_ordinary_sequence_dtypes(self):
        cases = [
            ([0, 1], np.dtype(np.int64)),
            ([0, np.nan, 1], np.dtype(np.float64)),
            ((-0.5, 1.5), np.dtype(np.float64)),
            (["a", "long"], np.dtype("U4")),
            ([b"a", b"long"], np.dtype("S4")),
            ([], np.dtype(np.float64)),
        ]
        for values, dtype in cases:
            with self.subTest(values=values):
                result = _as_label_array(values)
                self.assertEqual(result.dtype, dtype)
                np.testing.assert_array_equal(result, values)

    def test_preserves_integers_at_float_precision_boundary(self):
        cases = [
            ([2**53 - 1, 2**53], np.dtype(float)),
            ([2**53, 2**53 + 1], np.dtype(object)),
            ([-(2**53), -(2**53) - 1], np.dtype(object)),
            ([2**64 - 2, 2**64 - 1], np.dtype(object)),
            ([2**200, 2**200 + 1], np.dtype(object)),
        ]
        for labels, dtype in cases:
            with self.subTest(labels=repr(labels)):
                values = [*labels, np.nan]
                result = _as_label_array(values)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result[:2].tolist(), labels)
                self.assertTrue(np.isnan(result[-1]))
                self.assertEqual(values[:2], labels)

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
        for values in ([1, "missing"], ["a", None], [b"a", "b"]):
            with self.subTest(values=values):
                result = _as_label_array(values)
                self.assertEqual(result.dtype, np.dtype(object))
                self.assertEqual(result.tolist(), values)
        self.assertIs(_as_label_array(["a", None])[1], None)

    def test_reuses_existing_arrays_including_empty_shapes(self):
        for values in (
            np.array([0.0, np.nan]),
            np.array([2**53, 2**53 + 1], dtype=object),
            np.empty((0, 2)),
        ):
            with self.subTest(dtype=values.dtype, shape=values.shape):
                self.assertIs(_as_label_array(values), values)


class TestAsClassVocabularyArray(unittest.TestCase):
    def test_preserves_mixed_numpy_integer_scalars_exactly(self):
        classes = [np.int64(2**53), np.uint64(2**53 + 1)]
        result = _as_class_vocabulary_array(classes)
        self.assertEqual(result.dtype, np.dtype(np.int64))
        self.assertEqual(result.tolist(), [2**53, 2**53 + 1])

    def test_uses_unsigned_storage_when_required(self):
        classes = [np.int64(0), np.uint64(2**63)]
        result = _as_class_vocabulary_array(classes)
        self.assertEqual(result.dtype, np.dtype(np.uint64))
        self.assertEqual(result.tolist(), [0, 2**63])


class TestLosslessDecodeDtype(unittest.TestCase):
    def test_omitted_missing_label_differs_from_explicit_none(self):
        classes = [np.array([0, 1], dtype=np.int32)]
        self.assertEqual(_lossless_decode_dtype(classes), np.dtype(np.int32))
        self.assertEqual(
            _lossless_decode_dtype(classes, None), np.dtype(object)
        )
        self.assertEqual(
            _lossless_decode_dtype(classes, np.nan), np.dtype(float)
        )

    def test_preserves_integer_limits(self):
        cases = [
            (np.array([-(2**63)], dtype=np.int64), np.dtype(float)),
            (np.array([2**63 - 1], dtype=np.int64), np.dtype(object)),
            (np.array([2**63], dtype=np.uint64), np.dtype(float)),
            (np.array([2**64 - 1], dtype=np.uint64), np.dtype(object)),
        ]
        for classes, expected in cases:
            with self.subTest(classes=repr(classes)):
                self.assertEqual(
                    _lossless_decode_dtype([classes], np.nan), expected
                )

    def test_preserves_string_kinds_and_missing_label_length(self):
        self.assertEqual(
            _lossless_decode_dtype([np.array([b"a"]), np.array(["b"])]),
            np.dtype(object),
        )
        self.assertEqual(
            _lossless_decode_dtype([np.array(["a"])], "missing"),
            np.dtype("U7"),
        )
        self.assertEqual(
            _lossless_decode_dtype([np.array(["a"])], np.nan),
            np.dtype(object),
        )

    def test_complex_promotion_is_warning_free_and_lossless(self):
        cases = [
            (np.array([1 + 2j, 3 + 0j]), -1, np.dtype(complex)),
            (np.array([1 + 2j], dtype=np.complex64), -1.0, np.dtype(complex)),
            (np.array([1, 2]), complex(np.nan, 1), np.dtype(complex)),
            (np.array([2**53, 2**53 + 1]), -1 + 0j, np.dtype(object)),
        ]
        for classes, missing, expected in cases:
            with self.subTest(classes=repr(classes), missing=missing):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    self.assertEqual(
                        _lossless_decode_dtype([classes], missing), expected
                    )


class TestLabelFamily(unittest.TestCase):
    def test_dtype_decides_the_family_of_an_ordinary_array(self):
        cases = [
            (np.array([False, True]), "bool"),
            (np.array([0, 1], dtype=np.int8), "int"),
            (np.array([0, 1], dtype=np.uint64), "int"),
            (np.array([0.0, 1.5], dtype=np.float16), "float"),
            (np.array([0.0, 1.5]), "float"),
            (np.array(["cat", "dog"]), "str"),
            (np.array([[0, 1], [2, 3]]), "int"),
        ]
        for values, family in cases:
            with self.subTest(dtype=values.dtype):
                self.assertEqual(_label_family(values, name="y"), family)

    def test_object_arrays_are_scanned_value_by_value(self):
        cases = [
            ([np.bool_(True), False], "bool"),
            ([np.int8(0), 1, np.uint64(2)], "int"),
            ([np.float32(0.5), 1.5], "float"),
            ([np.str_("cat"), "dog"], "str"),
        ]
        for values, family in cases:
            with self.subTest(values=repr(values)):
                y = np.array(values, dtype=object)
                self.assertEqual(_label_family(y, name="y"), family)

    def test_lists_are_judged_by_their_array_representation(self):
        # `[0, 1.5]` becomes a floating-point array, whereas the equivalent
        # object array mixes an integer with a float.
        self.assertEqual(_label_family([0, 1.5], name="classes"), "float")
        with self.assertRaisesRegex(TypeError, "one label family"):
            _label_family(np.array([0, 1.5], dtype=object), name="classes")

    def test_empty_values_provide_no_evidence(self):
        for values in ([], np.array([]), np.empty((0, 2), dtype=object)):
            with self.subTest(values=repr(values)):
                self.assertIsNone(_label_family(values, name="y"))

    def test_rejects_dtypes_outside_the_contract(self):
        cases = [
            np.array([1 + 2j]),
            np.array([b"cat"]),
            np.array([1], dtype=np.longdouble),
            np.array(["2026-09-08"], dtype="M8[D]"),
            np.array([1], dtype="m8[s]"),
        ]
        for values in cases:
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(
                    TypeError, "unsupported label dtype"
                ):
                    _label_family(values, name="y")

    def test_rejects_scalar_types_outside_the_contract(self):
        from decimal import Decimal
        from fractions import Fraction

        cases = [
            1 + 2j,
            np.complex64(1),
            b"cat",
            Decimal("1.5"),
            Fraction(1, 2),
            np.array([0, 1]),
            [0, 1],
            {"label": 0},
        ]
        for value in cases:
            with self.subTest(value=repr(value)):
                y = np.empty(1, dtype=object)
                y[0] = value
                with self.assertRaisesRegex(
                    TypeError, "unsupported scalar label type"
                ):
                    _label_family(y, name="y")

    def test_rejects_extended_precision_float_scalars(self):
        y = np.array([np.longdouble(1.5)], dtype=object)
        with self.assertRaisesRegex(TypeError, "extended-precision float"):
            _label_family(y, name="y")

    def test_rejects_none_naming_the_missing_label(self):
        y = np.array([0, None], dtype=object)
        with self.assertRaisesRegex(TypeError, "only valid as"):
            _label_family(y, name="y")

    def test_integers_must_fit_a_64_bit_dtype(self):
        accepted = [-(2**63), -1, 0, 2**63 - 1, 2**63, 2**64 - 1]
        for value in accepted:
            with self.subTest(value=value):
                y = np.array([value], dtype=object)
                self.assertEqual(_label_family(y, name="y"), "int")
        for value in (-(2**63) - 1, 2**64, 2**200):
            with self.subTest(value=value):
                y = np.array([value], dtype=object)
                with self.assertRaisesRegex(
                    ValueError, "64-bit integer dtype"
                ):
                    _label_family(y, name="y")

    def test_integers_must_share_one_64_bit_dtype(self):
        y = np.array([-1, 2**64 - 1], dtype=object)
        with self.assertRaisesRegex(ValueError, "64-bit integer dtype"):
            _label_family(y, name="y")
        y = np.array([0, 2**64 - 1], dtype=object)
        self.assertEqual(_label_family(y, name="y"), "int")
        y = np.array([-1, 2**63 - 1], dtype=object)
        self.assertEqual(_label_family(y, name="y"), "int")

    def test_boolean_entries_do_not_constrain_the_integer_dtype(self):
        y = np.array([np.bool_(True), 2**64 - 1], dtype=object)
        with self.assertRaisesRegex(TypeError, "one label family"):
            _label_family(y, name="y")

    def test_rejects_nonfinite_values(self):
        for values in (
            np.array([0.0, np.nan]),
            np.array([0.0, np.nan], dtype=object),
        ):
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(ValueError, "contains NaN"):
                    _label_family(values, name="y")
        for values in (
            np.array([0.0, np.inf]),
            np.array([0.0, -np.inf], dtype=object),
        ):
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(
                    ValueError, "contains an infinite value"
                ):
                    _label_family(values, name="y")

    def test_mixed_families_name_every_family(self):
        y = np.array([True, 1, 1.5, "cat"], dtype=object)
        with self.assertRaises(TypeError) as caught:
            _label_family(y, name="classes")
        message = str(caught.exception)
        self.assertIn("`classes`", message)
        for description in ("Boolean", "integer", "floating-point", "string"):
            self.assertIn(description, message)

    def test_numerical_labels_collapse_the_numeric_families(self):
        cases = [
            np.array([0, 1]),
            np.array([0.0, 1.5]),
            np.array([0, 1.5], dtype=object),
        ]
        for values in cases:
            with self.subTest(values=repr(values)):
                self.assertEqual(
                    _label_family(values, name="y", role=_NUMERICAL_LABELS),
                    "float",
                )

    def test_numerical_labels_reject_boolean_and_string_values(self):
        for values in (np.array([True, False]), np.array(["0.5"])):
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(TypeError, "numerical labels"):
                    _label_family(values, name="y", role=_NUMERICAL_LABELS)

    def test_task_agnostic_labels_accept_either_task_domain(self):
        self.assertEqual(
            _label_family(
                np.array([0, 1.5], dtype=object),
                name="y",
                role=_TASK_AGNOSTIC_LABELS,
            ),
            "float",
        )
        for values in (
            np.array([False, 2], dtype=object),
            np.array([0, "one"], dtype=object),
        ):
            with self.subTest(values=repr(values)):
                with self.assertRaisesRegex(TypeError, "one label family"):
                    _label_family(values, name="y", role=_TASK_AGNOSTIC_LABELS)


class TestMissingLabelChecks(unittest.TestCase):
    def test_accepts_supported_missing_label_values(self):
        for missing_label in (
            None,
            "?",
            np.str_("?"),
            np.nan,
            np.float16(np.nan),
            -2.5,
            -1,
            np.int8(-1),
            np.uint64(2**63),
            2**64 - 1,
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
            np.longdouble(np.nan),
        ):
            with self.subTest(missing_label=repr(missing_label)):
                with self.assertRaisesRegex(TypeError, "must be a number"):
                    _check_missing_label_value(missing_label)

    def test_rejects_nonfinite_and_oversized_missing_labels(self):
        for missing_label in (np.inf, -np.inf, np.float32(np.inf)):
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    _check_missing_label_value(missing_label)
        with self.assertRaisesRegex(ValueError, "64-bit integer dtype"):
            _check_missing_label_value(2**64)

    def test_missing_label_compatibility_follows_the_contract(self):
        numeric_missing_labels = (np.nan, -2.5, -1)
        for family in ("bool", "int", "float"):
            with self.subTest(family=family):
                _check_missing_label_for_family(None, family, name="y")
                for missing_label in numeric_missing_labels:
                    _check_missing_label_for_family(
                        missing_label, family, name="y"
                    )
                with self.assertRaisesRegex(
                    TypeError, "is not compatible with"
                ):
                    _check_missing_label_for_family("?", family, name="y")
        _check_missing_label_for_family(None, "str", name="y")
        _check_missing_label_for_family("?", "str", name="y")
        for missing_label in numeric_missing_labels:
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(
                    TypeError, "is not compatible with"
                ):
                    _check_missing_label_for_family(
                        missing_label, "str", name="y"
                    )

    def test_unknown_family_accepts_every_missing_label(self):
        for missing_label in (None, "?", np.nan, -1):
            with self.subTest(missing_label=repr(missing_label)):
                _check_missing_label_for_family(missing_label, None, name="y")

    def test_missing_label_error_names_the_labels(self):
        with self.assertRaises(TypeError) as caught:
            _check_missing_label_for_family("?", "int", name="classes")
        self.assertIn("`classes`", str(caught.exception))
        with self.assertRaises(TypeError) as caught:
            _check_missing_label_for_family("?", "int")
        self.assertIn("target object", str(caught.exception))

    def test_numeric_nan_never_equals_the_string_missing_label(self):
        self.assertFalse(_matches_missing_label(np.nan, "nan"))
        self.assertFalse(_matches_missing_label("nan", np.nan))
        self.assertTrue(_matches_missing_label(np.nan, np.float32(np.nan)))
        self.assertTrue(_matches_missing_label(None, None))
        self.assertFalse(_matches_missing_label(None, np.nan))
        self.assertTrue(_matches_missing_label(-1, -1.0))
        self.assertFalse(_matches_missing_label(2**53 + 1, float(2**53)))

    def test_only_a_dtype_holding_the_missing_label_can_store_it(self):
        cases = [
            (np.array([0, 1]), np.nan, False),
            (np.array([0.0, 1.0]), np.nan, True),
            (np.array([0.0, 1.0]), None, False),
            (np.array([0.0, 1.0]), -1, True),
            (np.array(["cat", "dog"]), "?", True),
            (np.array([0, 1], dtype=object), None, True),
        ]
        for values, missing_label, expected in cases:
            with self.subTest(
                dtype=values.dtype, missing_label=repr(missing_label)
            ):
                self.assertEqual(
                    _holds_missing_label(values, missing_label), expected
                )


class TestMissingMaskAndFamily(unittest.TestCase):
    def test_missing_label_is_checked_against_the_dtype_of_an_array(self):
        # The dtype announces the family, so an incompatible missing label
        # is named as such instead of the missing values it fails to mark.
        y = np.array([0.0, np.nan, 1.0])
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            _missing_mask_and_family(y, "nan", name="y")

    def test_observed_values_of_an_object_array_decide_the_family(self):
        y = np.array(["cat", "?", "dog"], dtype=object)
        is_missing, family = _missing_mask_and_family(y, "?", name="y")
        np.testing.assert_array_equal(is_missing, [False, True, False])
        self.assertEqual(family, "str")
        # The missing label is checked against the observed values, not
        # against the object dtype, which announces no family.
        y = np.array(["cat", -1, "dog"], dtype=object)
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            _missing_mask_and_family(y, -1, name="y")

    def test_nan_among_observed_values_is_rejected(self):
        for y in (
            np.array([0.0, np.nan, 1.0]),
            np.array([0.0, np.nan, 1.0], dtype=object),
        ):
            with self.subTest(dtype=y.dtype):
                with self.assertRaisesRegex(ValueError, "contains NaN"):
                    _missing_mask_and_family(y, None, name="y")

    def test_missing_label_of_another_family_marks_nothing(self):
        y = np.array(["cat", "dog"])
        is_missing, family = _missing_mask_and_family(y, None, name="y")
        np.testing.assert_array_equal(is_missing, [False, False])
        self.assertEqual(family, "str")
        y = np.array([0, 1])
        is_missing, family = _missing_mask_and_family(y, np.nan, name="y")
        np.testing.assert_array_equal(is_missing, [False, False])
        self.assertEqual(family, "int")

    def test_entirely_missing_labels_provide_no_family_evidence(self):
        cases = [
            (np.array([np.nan, np.nan]), np.nan),
            (np.array([np.nan, np.nan], dtype=np.longdouble), np.nan),
            (np.array([-1, -1]), -1),
            (np.array(["?", "?"]), "?"),
            (np.array([None, None], dtype=object), None),
        ]
        for y, missing_label in cases:
            with self.subTest(dtype=y.dtype, missing_label=missing_label):
                is_missing, family = _missing_mask_and_family(
                    y, missing_label, name="y"
                )
                self.assertTrue(is_missing.all())
                self.assertIsNone(family)

    def test_numeric_missing_label_preserves_large_integer_identity(self):
        y = np.array([2**53, 2**53 + 1])
        is_missing, family = _missing_mask_and_family(
            y, float(2**53), name="y"
        )
        np.testing.assert_array_equal(is_missing, [True, False])
        self.assertEqual(family, "int")

    def test_regression_mode_enforces_the_missing_label_table(self):
        y = np.array([0, 1.5, np.nan])
        is_missing, family = _missing_mask_and_family(
            y, np.nan, name="y", role=_NUMERICAL_LABELS
        )
        np.testing.assert_array_equal(is_missing, [False, False, True])
        self.assertEqual(family, "float")
        y = np.array([0, 1.5, "?"], dtype=object)
        with self.assertRaisesRegex(TypeError, "is not compatible with"):
            _missing_mask_and_family(y, "?", name="y", role=_NUMERICAL_LABELS)

    def test_incompatible_missing_label_names_the_observed_family(self):
        # The numerical and task-agnostic roles decide compatibility for the
        # merged numeric family, but the message names what `y` holds.
        cases = [
            (np.array([1, 2]), _TASK_AGNOSTIC_LABELS, "integer"),
            (np.array([1, 2]), _NUMERICAL_LABELS, "integer"),
            (np.array([1.5, 2.5]), _NUMERICAL_LABELS, "floating-point"),
            (np.array([True, False]), _TASK_AGNOSTIC_LABELS, "Boolean"),
            (
                np.array([1, 2], dtype=object),
                _TASK_AGNOSTIC_LABELS,
                "integer",
            ),
            (
                np.array([1, 2.5], dtype=object),
                _TASK_AGNOSTIC_LABELS,
                "numerical",
            ),
            (
                np.array([1, 2.5], dtype=object),
                _NUMERICAL_LABELS,
                "numerical",
            ),
            (np.array([1, 2], dtype=object), _LABELS, "integer"),
        ]
        for y, role, description in cases:
            with self.subTest(y=repr(y), role=role):
                with self.assertRaisesRegex(
                    TypeError, f"the {description} labels in `y`"
                ):
                    _missing_mask_and_family(y, "?", name="y", role=role)

    def test_large_float_arrays_are_not_scanned_in_python(self):
        y = np.zeros(10**6)
        y[::2] = np.nan
        with unittest.mock.patch(
            "skactiveml.utils._label_dtype._scalar_label_family"
        ) as scanner:
            is_missing, family = _missing_mask_and_family(y, np.nan, name="y")
        scanner.assert_not_called()
        self.assertEqual(family, "float")
        self.assertEqual(is_missing.sum(), 5 * 10**5)

    def test_classification_labels_must_not_mix_numeric_families(self):
        cases = [
            np.array([0, 1.5, None], dtype=object),
            np.array([np.bool_(True), 1, None], dtype=object),
            np.array([np.bool_(True), 1.5, None], dtype=object),
        ]
        for y in cases:
            with self.subTest(y=repr(y)):
                with self.assertRaisesRegex(TypeError, "one label family"):
                    _missing_mask_and_family(y, None, name="y", role=_LABELS)

        is_missing, family = _missing_mask_and_family(
            np.array([np.bool_(True), None], dtype=object),
            None,
            name="y",
            role=_LABELS,
        )
        np.testing.assert_array_equal(is_missing, [False, True])
        self.assertEqual(family, "bool")

    def test_observed_labels_must_not_mix_numbers_and_strings(self):
        cases = [
            np.array(["cat", 1], dtype=object),
            np.array([np.bool_(True), "cat"], dtype=object),
            np.array([1.5, "cat"], dtype=object),
        ]
        for y in cases:
            with self.subTest(y=repr(y)):
                with self.assertRaisesRegex(TypeError, "one label family"):
                    _missing_mask_and_family(y, np.nan, name="y")
