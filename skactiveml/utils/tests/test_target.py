import unittest
from dataclasses import FrozenInstanceError, fields
from inspect import signature

import numpy as np

from skactiveml.utils import (
    TargetSpec,
    is_labeled,
    is_unlabeled,
    labeled_indices,
    resolve_target_spec,
    unlabeled_indices,
)
from skactiveml.utils._target import (
    _class_vocabulary_key,
    check_target_capability,
)


class TestTargetSpec(unittest.TestCase):
    def test_requires_classes_exactly_for_classification(self):
        with self.assertRaisesRegex(ValueError, "required for classification"):
            TargetSpec(
                task="classification",
                target_type="single-output",
                annotation_type="single-annotator",
                classes=None,
            )

        with self.assertRaisesRegex(ValueError, "not accepted for regression"):
            TargetSpec(
                task="regression",
                target_type="single-output",
                annotation_type="single-annotator",
                classes=(0, 1),
            )

    def test_rejects_incorrect_class_vocabulary_structure(self):
        invalid_specs = [
            ("single-output", (), "must not be empty"),
            ("single-output", (0, (0, 1)), "uniformly flat or nested"),
            ("single-output", ((0, 1), (0, 1)), "flat"),
            ("multi-label", (0, 1), "nested"),
            ("multi-label", ((0, 1, 2), (0, 1)), "exactly two"),
            ("multi-output", (0, 1), "nested"),
        ]

        for target_type, classes, message in invalid_specs:
            with self.subTest(target_type=target_type, classes=classes):
                with self.assertRaisesRegex(ValueError, message):
                    TargetSpec(
                        task="classification",
                        target_type=target_type,
                        annotation_type="single-annotator",
                        classes=classes,
                    )

    def test_canonicalizes_equivalent_direct_and_resolved_specs(self):
        direct_single_output = TargetSpec(
            task="classification",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=("dog", "cat"),
        )
        resolved_single_output = resolve_target_spec(
            ["dog", "cat"],
            task="classification",
            classes=("dog", "cat"),
            missing_label="missing",
        )
        direct_multi_label = TargetSpec(
            task="classification",
            target_type="multi-label",
            annotation_type="single-annotator",
            classes=(("yes", "no"), ("warm", "cold")),
        )
        resolved_multi_label = resolve_target_spec(
            [["yes", "warm"], ["no", "cold"]],
            task="classification",
            target_type="multi-label",
            classes=(("yes", "no"), ("warm", "cold")),
            missing_label="missing",
        )

        self.assertEqual(direct_single_output, resolved_single_output)
        self.assertEqual(direct_single_output.classes, ("cat", "dog"))
        self.assertEqual(direct_multi_label, resolved_multi_label)
        self.assertEqual(
            direct_multi_label.classes,
            (("no", "yes"), ("cold", "warm")),
        )

    def test_canonical_equality_treats_nan_classes_as_equal(self):
        direct = TargetSpec(
            task="classification",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=(np.nan, 1.0),
        )
        resolved = resolve_target_spec(
            [1.0],
            task="classification",
            classes=(np.nan, 1.0),
            missing_label=-1,
        )

        self.assertEqual(direct, resolved)
        self.assertEqual(hash(direct), hash(resolved))

    def test_equality_and_hashing_handle_empty_and_different_vocabularies(
        self,
    ):
        regression = TargetSpec(
            task="regression",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=None,
        )
        same_regression = TargetSpec(
            task="regression",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=None,
        )
        classification = TargetSpec(
            task="classification",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=(0, 1),
        )
        other_classification = TargetSpec(
            task="classification",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=(0, 1, 2),
        )

        self.assertEqual(regression, same_regression)
        self.assertEqual(hash(regression), hash(same_regression))
        self.assertNotEqual(classification, other_classification)
        self.assertFalse(classification == object())

    def test_rejects_unresolved_or_unknown_semantic_values(self):
        invalid_specs = [
            (
                {
                    "task": "clustering",
                    "target_type": "single-output",
                    "annotation_type": "single-annotator",
                },
                "task",
            ),
            (
                {
                    "task": "classification",
                    "target_type": "auto",
                    "annotation_type": "single-annotator",
                },
                "target_type",
            ),
            (
                {
                    "task": "classification",
                    "target_type": "multilabel",
                    "annotation_type": "single-annotator",
                },
                "target_type",
            ),
            (
                {
                    "task": "classification",
                    "target_type": "single-output",
                    "annotation_type": "crowd",
                },
                "annotation_type",
            ),
        ]

        for declarations, message in invalid_specs:
            with self.subTest(declarations=declarations):
                with self.assertRaisesRegex(ValueError, message):
                    TargetSpec(classes=None, **declarations)

    def test_rejects_incompatible_semantic_combinations(self):
        invalid_specs = [
            (
                {
                    "task": "regression",
                    "target_type": "multi-label",
                    "annotation_type": "single-annotator",
                },
                "requires classification",
            ),
            (
                {
                    "task": "classification",
                    "target_type": "multi-label",
                    "annotation_type": "multi-annotator",
                },
                "cannot be combined",
            ),
            (
                {
                    "task": "classification",
                    "target_type": "multi-output",
                    "annotation_type": "multi-annotator",
                },
                "cannot be combined",
            ),
        ]

        for declarations, message in invalid_specs:
            with self.subTest(declarations=declarations):
                with self.assertRaisesRegex(ValueError, message):
                    TargetSpec(classes=None, **declarations)

    def test_freezes_flat_and_nested_class_vocabularies(self):
        flat_classes = ["cat", "dog"]
        nested_classes = [np.array([0, 1]), [2, 3]]
        flat = TargetSpec(
            task="classification",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=flat_classes,
        )
        nested = TargetSpec(
            task="classification",
            target_type="multi-label",
            annotation_type="single-annotator",
            classes=nested_classes,
        )
        flat_classes.append("mouse")
        nested_classes[0][0] = 2
        nested_classes[1].append(4)

        self.assertEqual(flat.classes, ("cat", "dog"))
        self.assertEqual(nested.classes, ((0, 1), (2, 3)))
        with self.assertRaises(FrozenInstanceError):
            flat.classes = ("mouse",)


class TestResolveTargetSpec(unittest.TestCase):
    def test_rejects_empty_or_mixed_class_vocabulary_structure(self):
        for classes, message in (
            ([], "must not be empty"),
            ([0, [0, 1]], "uniformly flat or nested"),
        ):
            with self.subTest(classes=classes):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_target_spec(
                        [0, 1], task="classification", classes=classes
                    )

    def test_auto_classification_uses_unambiguous_declarations(self):
        single_output = resolve_target_spec(
            ["dog", "cat", "dog"],
            task="classification",
            classes=["dog", "cat"],
            missing_label="missing",
        )
        multi_label = resolve_target_spec(
            [[0, 1], [1, 0]],
            task="classification",
            classes=[[1, 0], [0, 1]],
        )
        multi_output = resolve_target_spec(
            [[0, 1], [1, 2]],
            task="classification",
            classes=[[1, 0], [2, 1, 0]],
        )

        self.assertEqual(single_output.target_type, "single-output")
        self.assertEqual(single_output.classes, ("cat", "dog"))
        self.assertEqual(multi_label.target_type, "multi-label")
        self.assertEqual(multi_label.classes, ((0, 1), (0, 1)))
        self.assertEqual(multi_output.target_type, "multi-output")
        self.assertEqual(
            multi_output.classes,
            ((0, 1), (0, 1, 2)),
        )

        inferred_single_output = resolve_target_spec(
            [0, 1], task="classification"
        )
        self.assertEqual(inferred_single_output.target_type, "single-output")

    def test_public_resolution_validates_shapes_and_metadata(self):
        with self.assertRaisesRegex(ValueError, "Nested class vocabularies"):
            resolve_target_spec(
                [[0, 1], [1, 0]],
                task="classification",
                annotation_type="multi-annotator",
                classes=[[0, 1], [0, 1]],
            )

        with self.assertRaisesRegex(ValueError, "No class label is observed"):
            resolve_target_spec(
                [np.nan, np.nan], task="classification", missing_label=np.nan
            )

        with self.assertRaisesRegex(TypeError, "one- or two-dimensional"):
            resolve_target_spec(1, task="regression")

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            resolve_target_spec(
                [0, 1],
                task="classification",
                annotation_type="multi-annotator",
                classes=[0, 1],
            )

        with self.assertRaisesRegex(
            ValueError, "one-dimensional or a column vector"
        ):
            resolve_target_spec(
                [[1.0, 2.0], [3.0, 4.0]],
                task="regression",
                target_type="single-output",
            )

        with self.assertRaisesRegex(ValueError, "No class label is observed"):
            resolve_target_spec(
                [[0, np.nan], [1, np.nan], [np.nan, np.nan]],
                task="classification",
                target_type="multi-output",
            )

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            resolve_target_spec(
                [0, 1], task="classification", target_type="multi-label"
            )

        with self.assertRaisesRegex(ValueError, "one vocabulary per"):
            resolve_target_spec(
                [[0, 1], [1, 0]],
                task="classification",
                target_type="multi-label",
                classes=[[0, 1]],
            )

    def test_public_resolution_reports_unknown_label_types_and_values(self):
        with self.assertRaisesRegex(TypeError, "not type-compatible"):
            resolve_target_spec(
                ["cat"],
                task="classification",
                classes=[0, 1],
                missing_label=None,
            )

        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_target_spec(
                [0, 2],
                task="classification",
                classes=[0, 1],
                missing_label=None,
            )

    def test_auto_classification_rejects_bare_two_dimensional_targets(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_target_spec(
                [[0, 1], [1, 0]],
                task="classification",
            )

    def test_auto_regression_resolves_single_and_future_multi_output(self):
        vector = resolve_target_spec([1.0, 2.0], task="regression")
        column = resolve_target_spec([[1.0], [2.0]], task="regression")
        matrix = resolve_target_spec(
            [[1.0, 2.0], [3.0, 4.0]], task="regression"
        )

        self.assertEqual(vector.target_type, "single-output")
        self.assertEqual(column.target_type, "single-output")
        self.assertEqual(matrix.target_type, "multi-output")
        self.assertIsNone(vector.classes)
        self.assertIsNone(column.classes)
        self.assertIsNone(matrix.classes)

    def test_explicit_regression_target_types_match_declared_structure(self):
        single_output = resolve_target_spec(
            [[1.0], [2.0]],
            task="regression",
            target_type="single-output",
        )
        multi_output = resolve_target_spec(
            [[1.0, 2.0], [3.0, 4.0]],
            task="regression",
            target_type="multi-output",
        )

        self.assertEqual(single_output.target_type, "single-output")
        self.assertEqual(multi_output.target_type, "multi-output")

    def test_empty_targets_resolve_when_metadata_is_sufficient(self):
        single_output = resolve_target_spec(
            [], task="classification", classes=[1, 0]
        )
        multi_label = resolve_target_spec(
            np.empty((0, 2)),
            task="classification",
            classes=[[1, 0], [3, 2]],
        )
        regression = resolve_target_spec([], task="regression")

        self.assertEqual(single_output.classes, (0, 1))
        self.assertEqual(multi_label.target_type, "multi-label")
        self.assertEqual(multi_label.classes, ((0, 1), (2, 3)))
        self.assertEqual(regression.target_type, "single-output")

    def test_multi_annotator_declaration_preserves_elementwise_targets(self):
        target_spec = resolve_target_spec(
            [[0, np.nan], [np.nan, 1]],
            task="classification",
            annotation_type="multi-annotator",
            classes=[1, 0],
        )

        self.assertEqual(
            target_spec,
            TargetSpec(
                task="classification",
                target_type="single-output",
                annotation_type="multi-annotator",
                classes=(0, 1),
            ),
        )

    def test_semantically_invalid_declarations_are_not_capability_errors(self):
        invalid_cases = [
            ([[0, 1], [1, 0]], {"task": "clustering"}, "task"),
            (
                [[0, 1], [1, 0]],
                {"task": "classification", "target_type": "multilabel"},
                "target_type",
            ),
            (
                [[0, 1], [1, 0]],
                {"task": "classification", "annotation_type": "crowd"},
                "annotation_type",
            ),
            (
                [[0, 1], [1, 0]],
                {"task": "regression", "target_type": "multi-label"},
                "requires classification",
            ),
            (
                [[0, 1], [1, 0]],
                {
                    "task": "classification",
                    "target_type": "single-output",
                },
                "one-dimensional",
            ),
            (
                [0, 1],
                {
                    "task": "classification",
                    "target_type": "multi-output",
                },
                "at least two",
            ),
            (
                [[0, 1], [1, 0]],
                {
                    "task": "classification",
                    "target_type": "multi-label",
                    "annotation_type": "multi-annotator",
                },
                "cannot be combined",
            ),
        ]

        for y, declarations, message in invalid_cases:
            with self.subTest(declarations=declarations):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_target_spec(
                        y,
                        classes=None,
                        **declarations,
                    )

        with self.assertRaisesRegex(ValueError, "not accepted for regression"):
            resolve_target_spec(
                [1.0, 2.0], task="regression", classes=[1.0, 2.0]
            )

    def test_declared_structure_must_match_target_columns(self):
        with self.assertRaisesRegex(ValueError, "one vocabulary per target"):
            resolve_target_spec(
                [[0, 1], [1, 0]],
                task="classification",
                target_type="multi-output",
                classes=[[0, 1]],
            )

        with self.assertRaisesRegex(ValueError, "nested"):
            resolve_target_spec(
                [[0, 1], [1, 0]],
                task="classification",
                target_type="multi-output",
                classes=[0, 1],
            )

        with self.assertRaisesRegex(ValueError, "nested binary"):
            resolve_target_spec(
                [[0, 1], [1, 0]],
                task="classification",
                target_type="multi-label",
                classes=[0, 1],
            )

    def test_explicit_multilabel_infers_immutable_canonical_spec(self):
        y = np.array(
            [
                ["yes", "cold"],
                ["no", "warm"],
                ["yes", "warm"],
                ["no", "cold"],
            ]
        )

        target_spec = resolve_target_spec(
            y,
            task="classification",
            target_type="multi-label",
            classes=None,
            missing_label="missing",
        )

        self.assertEqual(
            [field.name for field in fields(TargetSpec)],
            ["task", "target_type", "annotation_type", "classes"],
        )
        self.assertEqual(
            target_spec,
            TargetSpec(
                task="classification",
                target_type="multi-label",
                annotation_type="single-annotator",
                classes=(("no", "yes"), ("cold", "warm")),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            target_spec.target_type = "auto"

    def test_reordered_multilabel_declarations_resolve_equally(self):
        y = np.array([["yes", "cold"], ["no", "warm"]])

        forward = resolve_target_spec(
            y,
            task="classification",
            target_type="multi-label",
            classes=(("no", "yes"), ("cold", "warm")),
            missing_label="missing",
        )
        reversed_declarations = resolve_target_spec(
            y,
            task="classification",
            target_type="multi-label",
            classes=(("yes", "no"), ("warm", "cold")),
            missing_label="missing",
        )

        self.assertEqual(forward, reversed_declarations)

    def test_undeclared_vocabularies_have_no_comparison_key(self):
        self.assertIsNone(_class_vocabulary_key(None))

    def test_multilabel_rejects_partially_observed_rows(self):
        y = np.array([[0, 1], [1, np.nan], [np.nan, np.nan]])

        with self.assertRaisesRegex(ValueError, "no mixing within a row"):
            resolve_target_spec(
                y,
                task="classification",
                target_type="multi-label",
            )

    def test_multilabel_without_classes_requires_two_observed_classes(self):
        y = np.array([[0, 1], [0, 0], [np.nan, np.nan]])

        with self.assertRaisesRegex(ValueError, "output 0 exposes 1"):
            resolve_target_spec(
                y,
                task="classification",
                target_type="multi-label",
            )

    def test_multilabel_without_classes_rejects_wholly_missing_output(self):
        y = np.empty((0, 2))

        with self.assertRaisesRegex(ValueError, "output 0 exposes 0"):
            resolve_target_spec(
                y,
                task="classification",
                target_type="multi-label",
            )

    def test_nested_binary_classes_allow_under_observed_outputs(self):
        target_spec = resolve_target_spec(
            [[0, 1], [0, 1], [np.nan, np.nan]],
            task="classification",
            target_type="multi-label",
            classes=[[0, 1], [0, 1]],
        )

        self.assertEqual(target_spec.classes, ((0, 1), (0, 1)))

    def test_valid_future_semantics_fail_only_at_capability_boundary(self):
        target_spec = resolve_target_spec(
            [[0, 0], [1, 2]],
            task="classification",
            classes=[[0, 1], [0, 1, 2]],
        )
        capabilities = frozenset(
            {("classification", "single-output", "single-annotator")}
        )

        with self.assertRaisesRegex(
            ValueError,
            "ExampleComponent.*multi-output.*Supported "
            "capabilities.*single-output",
        ):
            check_target_capability(
                "ExampleComponent", target_spec, capabilities
            )

    def test_multilabel_label_helpers_return_sample_level_results(self):
        y = np.array([[0, 1], [np.nan, np.nan], [1, 0]])

        np.testing.assert_array_equal(
            is_labeled(y, target_type="multi-label"), [True, False, True]
        )
        np.testing.assert_array_equal(
            is_unlabeled(y, target_type="multi-label"), [False, True, False]
        )
        np.testing.assert_array_equal(
            labeled_indices(y, target_type="multi-label"), [0, 2]
        )
        np.testing.assert_array_equal(
            unlabeled_indices(y, target_type="multi-label"), [1]
        )

    def test_label_helpers_reject_unresolved_target_type(self):
        with self.assertRaisesRegex(ValueError, "resolved target type"):
            is_unlabeled([0, np.nan], target_type="auto")

    def test_label_helper_target_type_defaults_are_keyword_only(self):
        for helper in (
            is_labeled,
            is_unlabeled,
            labeled_indices,
            unlabeled_indices,
        ):
            parameter = signature(helper).parameters["target_type"]
            self.assertEqual(parameter.default, "single-output")
            self.assertEqual(parameter.kind.name, "KEYWORD_ONLY")

        y = np.array([[0.0, np.nan], [np.nan, 1.0]])
        np.testing.assert_array_equal(
            is_labeled(y), [[True, False], [False, True]]
        )
        np.testing.assert_array_equal(labeled_indices(y), [[0, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
