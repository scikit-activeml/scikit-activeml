"""Inventories of the multi-label contracts of pool strategies.

Strategies consuming multi-label probabilities must accept both
`proba_format` values of `SklearnClassifier`, i.e., `"array"` and `"list"`.
The inventories below keep the shared template tests of
`TemplatePoolQueryStrategy` applied to every multi-label capable strategy.
"""

import importlib
import inspect
import pkgutil
import re
import unittest
from copy import deepcopy
from functools import cache
from pathlib import Path
from unittest.mock import patch

import numpy as np

import skactiveml.pool
import skactiveml.pool.tests
from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import (
    Badge,
    Clue,
    CoreSet,
    DiscriminativeAL,
    DropQuery,
    Falcun,
    GreedySamplingX,
    LabelCardinalityInconsistency,
    MaxHerding,
    MaxLossReductionMaxConfidence,
    ParallelUtilityEstimationWrapper,
    ProbCover,
    RandomSampling,
    SubSamplingWrapper,
    TypiClust,
    UHerding,
    UncertaintySampling,
)
from skactiveml.tests.template_query_strategy import TemplatePoolQueryStrategy
from skactiveml.pool.tests._strategy_helpers import _instantiate

MULTILABEL_CAPABILITY = ("classification", "multi-label", "single-annotator")


@cache
def _collect_pool_template_test_cases():
    """Collect initialized pool test cases that reuse the shared template."""
    test_cases = []
    for module_info in pkgutil.iter_modules(skactiveml.pool.tests.__path__):
        module = importlib.import_module(
            f"{skactiveml.pool.tests.__name__}.{module_info.name}"
        )
        for _, test_class in inspect.getmembers(module, inspect.isclass):
            if (
                not issubclass(test_class, TemplatePoolQueryStrategy)
                or not issubclass(test_class, unittest.TestCase)
                or test_class.__module__ != module.__name__
            ):
                continue
            test_case = test_class("runTest")
            test_case.setUp()
            test_cases.append(test_case)
    return tuple(test_cases)


# Strategies consuming multi-label probabilities of the given classifier.
MULTILABEL_PROBA_CONSUMERS = frozenset(
    {
        Badge,
        Clue,
        Falcun,
        MaxLossReductionMaxConfidence,
        UHerding,
        UncertaintySampling,
    }
)

# Strategies consuming multi-label predictions only.
MULTILABEL_PREDICTION_CONSUMERS = frozenset(
    {
        DropQuery,
        LabelCardinalityInconsistency,
    }
)

# Strategies using only the sample representations and the label mask.
MULTILABEL_ESTIMATOR_FREE = frozenset(
    {
        CoreSet,
        DiscriminativeAL,
        GreedySamplingX,
        MaxHerding,
        ProbCover,
        RandomSampling,
        TypiClust,
    }
)

# Wrappers inheriting their target capabilities from the wrapped strategy.
MULTILABEL_DELEGATING_WRAPPERS = frozenset(
    {
        ParallelUtilityEstimationWrapper,
        SubSamplingWrapper,
    }
)


def _multilabel_capable_strategies():
    """Collect the pool strategies declaring multi-label support."""
    strategies = set()
    for name in dir(skactiveml.pool):
        obj = getattr(skactiveml.pool, name)
        if not inspect.isclass(obj) or not issubclass(
            obj, SingleAnnotatorPoolQueryStrategy
        ):
            continue
        if MULTILABEL_CAPABILITY in _instantiate(obj)._target_capabilities:
            strategies.add(obj)
    return strategies


def _multilabel_inventory():
    """Returns every strategy classified by the enforced inventory."""
    return (
        MULTILABEL_PROBA_CONSUMERS
        | MULTILABEL_PREDICTION_CONSUMERS
        | MULTILABEL_ESTIMATOR_FREE
        | MULTILABEL_DELEGATING_WRAPPERS
    )


def _multilabel_test_cases_by_strategy():
    """Collects the test cases reusing the shared pool strategy template."""
    test_cases = {}
    for test_case in _collect_pool_template_test_cases():
        test_cases.setdefault(test_case.qs_class, []).append(test_case)
    return test_cases


class TestMultilabelProbaFormatContract(unittest.TestCase):
    def test_target_semantics_inventory_matches_capabilities(self):
        target_semantics = (
            Path(__file__).parents[3] / "docs" / "target_semantics.rst"
        ).read_text()
        marker = ".. _multilabel-strategy-inventory:"
        end_marker = "Estimator Capability for Multi-label Wrapping"
        self.assertIn(marker, target_semantics)
        self.assertIn(end_marker, target_semantics)
        inventory_section = target_semantics.split(marker, maxsplit=1)[
            1
        ].split(end_marker, maxsplit=1)[0]
        documented_strategies = set(
            re.findall(
                r":class:`~skactiveml\.pool\.([A-Za-z0-9_]+)`",
                inventory_section,
            )
        )

        self.assertEqual(
            {strategy.__name__ for strategy in _multilabel_inventory()},
            documented_strategies,
            msg="The target-semantics reference must categorize every "
            "multi-label-capable pool strategy from this enforced inventory.",
        )

    def test_multilabel_capable_strategies_are_inventoried(self):
        self.assertEqual(
            {
                strategy.__name__
                for strategy in _multilabel_capable_strategies()
            },
            {strategy.__name__ for strategy in _multilabel_inventory()},
            msg="A multi-label-capable pool strategy is missing from the "
            "inventory. Classify it as a probability consumer, a prediction "
            "consumer, a task-agnostic strategy, or a delegating wrapper.",
        )

    def test_multilabel_proba_consumers_reuse_shared_contract_test(self):
        test_cases = _multilabel_test_cases_by_strategy()

        for strategy in MULTILABEL_PROBA_CONSUMERS:
            with self.subTest(strategy=strategy.__name__):
                strategy_test_cases = test_cases.get(strategy, [])
                self.assertTrue(
                    strategy_test_cases,
                    msg=f"{strategy.__name__} has no test case reusing "
                    f"`TemplatePoolQueryStrategy`.",
                )
                for test_case in strategy_test_cases:
                    query_params = (
                        test_case.query_default_params_clf_multilabel
                    )
                    self.assertIsNotNone(
                        query_params,
                        msg=f"{strategy.__name__}'s test case defines no "
                        f"`query_default_params_clf_multilabel`, so the "
                        f"shared probability-format contract test is "
                        f"skipped.",
                    )
                    self.assertIsNotNone(
                        test_case._multilabel_proba_format_estimator_key(
                            query_params
                        ),
                        msg=f"{strategy.__name__}'s multi-label test "
                        f"parameters contain no `SklearnClassifier` whose "
                        f"`proba_format` can be varied.",
                    )

    def test_multilabel_inventory_matches_probability_consumption(self):
        test_cases = _multilabel_test_cases_by_strategy()
        non_consumers = (
            MULTILABEL_PREDICTION_CONSUMERS | MULTILABEL_ESTIMATOR_FREE
        )

        for strategy, consumes_probas in [
            *((strategy, True) for strategy in MULTILABEL_PROBA_CONSUMERS),
            *((strategy, False) for strategy in non_consumers),
        ]:
            for test_case in test_cases.get(strategy, []):
                query_params = test_case.query_default_params_clf_multilabel
                if query_params is None:
                    continue
                estimator_key = (
                    test_case._multilabel_proba_format_estimator_key(
                        query_params
                    )
                )
                if estimator_key is None:
                    continue

                with self.subTest(strategy=strategy.__name__):
                    query_params = deepcopy(query_params)
                    n_outputs = np.asarray(query_params["y"]).shape[1]
                    malformed = [
                        np.full((len(query_params["X"]), 2), 0.5)
                        for _ in range(n_outputs + 1)
                    ]
                    qs = test_case.qs_class(
                        **test_case._multilabel_init_params()
                    )
                    with patch.object(
                        SklearnClassifier,
                        "predict_proba",
                        return_value=malformed,
                    ):
                        if consumes_probas:
                            with self.assertRaisesRegex(
                                ValueError,
                                f"outputs, expected {n_outputs}",
                                msg=f"{strategy.__name__} is inventoried as "
                                f"a probability consumer but accepts a "
                                f"malformed output count.",
                            ):
                                qs.query(**query_params)
                        else:
                            qs.query(**query_params)


# Test cases relabeling their default multi-label query parameters.
DEFAULT_VOCABULARY_FIXTURE = frozenset(
    {
        "TestBadge",
        "TestClue",
        "TestCoreSet",
        "TestDiscriminativeAL",
        "TestDropQuery",
        "TestFalcun",
        "TestGreedySamplingX",
        "TestLabelCardinalityInconsistency",
        "TestMaxHerding",
        "TestMaxLossReductionMaxConfidence",
        "TestParallelUtilityEstimationWrapper",
        "TestProbCover",
        "TestRandomSampling",
        "TestSubSamplingWrapper",
        "TestTypiClust",
        "TestUncertaintySampling",
    }
)

# Test cases providing their own fixture, with the reason.
SPECIALIZED_VOCABULARY_FIXTURE = {
    "TestUHerding": "needs logits and a multi-element temperature grid to "
    "reach per-output calibration",
}

# Shared tests of `TemplatePoolQueryStrategy` that no test case may replace.
CONTRACT_TEST_NAMES = (
    "test_query_multilabel_custom_class_vocabularies",
    "test_query_multilabel_invalid_rows",
)


def _multilabel_test_cases_with_params():
    """Collect the pool test cases defining multi-label query parameters."""
    return [
        test_case
        for test_case in _collect_pool_template_test_cases()
        if test_case.query_default_params_clf_multilabel is not None
    ]


def _overrides_fixture(test_case):
    return (
        type(test_case)._multilabel_custom_vocabulary_params
        is not TemplatePoolQueryStrategy._multilabel_custom_vocabulary_params
    )


class _AnyOutputCountTestCase(TemplatePoolQueryStrategy):
    """Test case standing in for a fixture with an arbitrary output count."""

    def __init__(self, n_outputs):
        self.query_default_params_clf_multilabel = {
            "y": np.zeros((2, n_outputs))
        }


class TestMultilabelVocabularyContract(unittest.TestCase):
    def test_string_vocabularies_are_distinct_and_canonical(self):
        for n_outputs in range(1, 5):
            with self.subTest(n_outputs=n_outputs):
                vocabularies = _AnyOutputCountTestCase(
                    n_outputs
                )._multilabel_string_vocabularies()

                self.assertEqual(len(vocabularies), n_outputs)
                self.assertEqual(len(set(vocabularies)), n_outputs)
                for negative, positive in vocabularies:
                    self.assertLess(negative, positive)

    def test_multilabel_test_cases_are_inventoried(self):
        inventory = DEFAULT_VOCABULARY_FIXTURE | set(
            SPECIALIZED_VOCABULARY_FIXTURE
        )

        self.assertEqual(
            {
                type(test_case).__name__
                for test_case in _multilabel_test_cases_with_params()
            },
            inventory,
            msg="A multi-label pool strategy test is missing from the "
            "custom class-vocabulary inventory. Record it as using the "
            "default fixture or as providing a specialized override.",
        )

    def test_inventory_matches_the_used_fixture(self):
        for test_case in _multilabel_test_cases_with_params():
            name = type(test_case).__name__
            with self.subTest(test_case=name):
                if name in SPECIALIZED_VOCABULARY_FIXTURE:
                    self.assertTrue(
                        _overrides_fixture(test_case),
                        msg=f"{name} is inventoried as providing a "
                        f"specialized custom-vocabulary fixture but uses the "
                        f"default one.",
                    )
                else:
                    self.assertFalse(
                        _overrides_fixture(test_case),
                        msg=f"{name} overrides "
                        f"`_multilabel_custom_vocabulary_params`. Record the "
                        f"override and its reason in "
                        f"`SPECIALIZED_VOCABULARY_FIXTURE`.",
                    )

    def test_contract_tests_are_not_replaced(self):
        for test_case in _multilabel_test_cases_with_params():
            for test_name in CONTRACT_TEST_NAMES:
                with self.subTest(
                    test_case=type(test_case).__name__, test=test_name
                ):
                    self.assertIs(
                        getattr(type(test_case), test_name),
                        getattr(TemplatePoolQueryStrategy, test_name),
                        msg=f"{type(test_case).__name__} replaces the shared "
                        f"contract test `{test_name}`. Customize the fixture "
                        f"through `_multilabel_custom_vocabulary_params` "
                        f"instead.",
                    )
