import unittest

import numpy as np
from sklearn.exceptions import DataConversionWarning

from skactiveml.base import _TaskAgnosticPoolQueryStrategy
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import (
    CoreSet,
    DiscriminativeAL,
    GreedySamplingX,
    MaxHerding,
    ProbCover,
    RandomSampling,
    TypiClust,
)
from skactiveml.tests.utils import assert_no_query_state

TASK_AGNOSTIC_CAPABILITIES = frozenset(
    {
        ("classification", "single-output", "single-annotator"),
        ("classification", "multi-label", "single-annotator"),
        ("regression", "single-output", "single-annotator"),
    }
)


def _strategy_cases(target_type="auto"):
    common = {
        "missing_label": -1,
        "random_state": 0,
        "target_type": target_type,
    }
    return (
        (RandomSampling(**common), {}),
        (CoreSet(**common), {}),
        (
            TypiClust(
                **common,
                cluster_algo_dict={"n_init": 1},
                k=2,
            ),
            {},
        ),
        (
            ProbCover(
                **common,
                deltas=[0.2],
                cluster_algo_dict={"n_init": 1},
            ),
            {},
        ),
        (MaxHerding(**common, metric="linear"), {}),
        (
            DiscriminativeAL(**common, greedy_selection=True),
            {"discriminator": ParzenWindowClassifier(random_state=0)},
        ),
        (GreedySamplingX(**common), {}),
    )


class TestTaskAgnosticTargetSemantics(unittest.TestCase):
    def test_public_target_type_and_exact_capabilities(self):
        strategy_cases = _strategy_cases()
        self.assertEqual(
            {type(strategy) for strategy, _ in strategy_cases},
            set(_TaskAgnosticPoolQueryStrategy.__subclasses__()),
            msg="Every task-agnostic strategy must have a behavioral case.",
        )

        for strategy, _ in strategy_cases:
            with self.subTest(strategy=type(strategy).__name__):
                self.assertEqual(strategy.target_type, "auto")
                self.assertEqual(
                    strategy._target_capabilities,
                    TASK_AGNOSTIC_CAPABILITIES,
                )

    def test_numeric_target_dtype_does_not_change_acquisition(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        y_integer = np.array([0, 1, -1, -1, -1, -1, -1, -1])
        y_float = y_integer.astype(float)

        integer_cases = _strategy_cases()
        float_cases = _strategy_cases()
        for (integer_strategy, integer_extra), (
            float_strategy,
            float_extra,
        ) in zip(integer_cases, float_cases):
            with self.subTest(strategy=type(integer_strategy).__name__):
                integer_result = integer_strategy.query(
                    X,
                    y_integer,
                    batch_size=2,
                    return_utilities=True,
                    **integer_extra,
                )
                float_result = float_strategy.query(
                    X,
                    y_float,
                    batch_size=2,
                    return_utilities=True,
                    **float_extra,
                )
                np.testing.assert_array_equal(
                    integer_result[0], float_result[0]
                )
                np.testing.assert_allclose(
                    integer_result[1], float_result[1], equal_nan=True
                )

    def test_explicit_multilabel_uses_sample_level_masks(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        y_single = np.array([0, 1, -1, -1, -1, -1, -1, -1])
        y_multilabel = np.array(
            [
                [0, 1],
                [1, 0],
                *[[-1, -1] for _ in range(6)],
            ]
        )

        single_cases = _strategy_cases(target_type="single-output")
        multilabel_cases = _strategy_cases(target_type="multi-label")
        for (single_strategy, single_extra), (
            multilabel_strategy,
            multilabel_extra,
        ) in zip(single_cases, multilabel_cases):
            with self.subTest(strategy=type(single_strategy).__name__):
                single_result = single_strategy.query(
                    X,
                    y_single,
                    batch_size=2,
                    return_utilities=True,
                    **single_extra,
                )
                multilabel_result = multilabel_strategy.query(
                    X,
                    y_multilabel,
                    batch_size=2,
                    return_utilities=True,
                    **multilabel_extra,
                )
                np.testing.assert_array_equal(
                    single_result[0], multilabel_result[0]
                )
                np.testing.assert_allclose(
                    single_result[1],
                    multilabel_result[1],
                    equal_nan=True,
                )

    def test_auto_rejects_ambiguous_matrix_before_query_state(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        y = np.array(
            [
                [0, 1],
                [1, 0],
                *[[-1, -1] for _ in range(6)],
            ]
        )
        strategy_specific_state_attributes = {
            "delta_max_",
            "distances_",
        }

        for strategy, extra in _strategy_cases():
            with self.subTest(strategy=type(strategy).__name__):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    strategy.query(X, y, **extra)
                assert_no_query_state(self, strategy)
                self.assertTrue(
                    strategy_specific_state_attributes.isdisjoint(
                        strategy.__dict__
                    )
                )

    def test_semantic_and_capability_failures_precede_query_state(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        invalid_cases = (
            (
                "invalid",
                np.array([0, 1, -1, -1, -1, -1, -1, -1]),
                "target_type",
            ),
            (
                "multi-output",
                np.array(
                    [
                        [0, 2],
                        [1, 3],
                        *[[-1, -1] for _ in range(6)],
                    ]
                ),
                "does not support target capability",
            ),
            (
                "multi-label",
                np.array(
                    [
                        [0, 1],
                        [1, 0],
                        [-1, 0],
                        *[[-1, -1] for _ in range(5)],
                    ]
                ),
                "mixing within a row",
            ),
        )

        for target_type, y, message in invalid_cases:
            for strategy, extra in _strategy_cases(target_type=target_type):
                with self.subTest(
                    strategy=type(strategy).__name__, target_type=target_type
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        strategy.query(X, y, **extra)
                    assert_no_query_state(self, strategy)

    def test_target_types_validate_public_shapes(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y_column = np.array([[0], [1], [-1], [-1]])

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            RandomSampling(
                target_type="auto", missing_label=-1, random_state=0
            ).query(X, y_column, batch_size=1)

        with self.assertWarns(DataConversionWarning):
            result = RandomSampling(
                target_type="single-output", missing_label=-1, random_state=0
            ).query(
                X,
                y_column,
                batch_size=1,
            )
        self.assertEqual(result.shape, (1,))

        invalid_cases = (
            (
                "single-output",
                np.array([[0, 1], [1, 0], [-1, -1], [-1, -1]]),
                "one-dimensional or a column",
            ),
            ("multi-label", np.array([0, 1, -1, -1]), "two-dimensional"),
            ("multi-output", np.array([[0], [1], [-1], [-1]]), "at least two"),
        )
        for target_type, y, message in invalid_cases:
            with self.subTest(target_type=target_type):
                with self.assertRaisesRegex(ValueError, message):
                    RandomSampling(
                        target_type=target_type,
                        missing_label=-1,
                        random_state=0,
                    ).query(X, y, batch_size=1)
