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
from skactiveml.utils._target import check_target_capability


class TestTargetSpec(unittest.TestCase):
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
        flat = TargetSpec(
            task="classification",
            target_type="single-output",
            annotation_type="single-annotator",
            classes=["cat", "dog"],
        )
        nested = TargetSpec(
            task="classification",
            target_type="multi-label",
            annotation_type="single-annotator",
            classes=[np.array([0, 1]), [2, 3]],
        )

        self.assertEqual(flat.classes, ("cat", "dog"))
        self.assertEqual(nested.classes, ((0, 1), (2, 3)))


class TestResolveTargetSpec(unittest.TestCase):
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
