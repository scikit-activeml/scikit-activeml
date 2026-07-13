import unittest
from dataclasses import FrozenInstanceError, fields

import numpy as np

from skactiveml.utils import (
    TargetSpec,
    is_labeled,
    is_unlabeled,
    labeled_indices,
    resolve_target_spec,
    unlabeled_indices,
)


class TestResolveTargetSpec(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
