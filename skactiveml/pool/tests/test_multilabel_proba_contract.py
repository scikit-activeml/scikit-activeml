"""Inventory of the multilabel probability-format contract of pool strategies.

`SklearnClassifier.proba_format` decides which of the two public multilabel
probability representations a query strategy observes:

- `"array"` yields one `(n_samples, n_outputs)` matrix of positive-class
  probabilities, and
- `"list"` yields one aligned `(n_samples, 2)` matrix per output.

This representation is independent of the native output of the wrapped
estimator, e.g. `MultiOutputClassifier` returns a list natively while
`OneVsRestClassifier` returns an array. Every strategy that declares
multilabel support and consumes probabilities must therefore canonicalize
them via `_canonicalize_multilabel_probas` before any indexing, sorting,
distance, or utility arithmetic.

The tests below keep the inventory of multilabel-capable pool strategies
complete and assert that each probability consumer is covered by the shared
contract test `test_query_multilabel_proba_format_contract` of
`TemplatePoolQueryStrategy`.

A strategy added to the inventory below must also be classified in
`docs/adr/0002-document-non-native-multilabel-acquisition.md`, which records
whether its multilabel behavior extends a single-output method and therefore
has to be declared in its docstring. The two partitions do not coincide: the
buckets below separate strategies by probability consumption, while the ADR
separates them by whether the cited method covers multilabel targets natively.
"""

import inspect
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

import skactiveml.pool
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
from skactiveml.pool.tests._multilabel_target_semantics import (
    collect_pool_template_test_cases,
)

MULTILABEL_CAPABILITY = ("classification", "multi-label", "single-annotator")

# Strategies consuming multilabel probabilities of the given classifier. They
# must accept both public probability formats.
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

# Strategies consuming multilabel predictions only, i.e., they are unaffected
# by the probability format.
MULTILABEL_PREDICTION_CONSUMERS = frozenset(
    {
        DropQuery,
        LabelCardinalityInconsistency,
    }
)

# Task-agnostic strategies operating on the sample representations and the
# label mask only, i.e., they never call `predict_proba` of a multilabel
# classifier.
MULTILABEL_TASK_AGNOSTIC = frozenset(
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

# Wrappers inheriting their target capabilities, and therefore this contract,
# from the wrapped query strategy.
MULTILABEL_DELEGATING_WRAPPERS = frozenset(
    {
        ParallelUtilityEstimationWrapper,
        SubSamplingWrapper,
    }
)


def _instantiate(strategy):
    """Instantiates a strategy by defaulting its required arguments."""
    init_params = {}
    for name, parameter in inspect.signature(
        strategy.__init__
    ).parameters.items():
        if name == "self":
            continue
        if name == "query_strategy":
            # Wrappers delegate their capabilities to the wrapped strategy,
            # so they must not be inspected with an absent one.
            init_params[name] = RandomSampling()
        elif parameter.default is inspect.Parameter.empty:
            init_params[name] = None
    return strategy(**init_params)


def _multilabel_capable_strategies():
    """Collects the pool strategies declaring multilabel support."""
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


def _multilabel_test_cases():
    """Collects the test cases reusing the shared pool strategy template."""
    test_cases = {}
    for test_case in collect_pool_template_test_cases():
        test_cases.setdefault(test_case.qs_class, []).append(test_case)
    return test_cases


class TestMultilabelProbaFormatContract(unittest.TestCase):
    def test_multilabel_capable_strategies_are_inventoried(self):
        inventory = (
            MULTILABEL_PROBA_CONSUMERS
            | MULTILABEL_PREDICTION_CONSUMERS
            | MULTILABEL_TASK_AGNOSTIC
            | MULTILABEL_DELEGATING_WRAPPERS
        )

        self.assertEqual(
            {
                strategy.__name__
                for strategy in _multilabel_capable_strategies()
            },
            {strategy.__name__ for strategy in inventory},
            msg="A multilabel-capable pool strategy is missing from the "
            "inventory. Classify it as a probability consumer, a prediction "
            "consumer, a task-agnostic strategy, or a delegating wrapper.",
        )

    def test_multilabel_proba_consumers_reuse_shared_contract_test(self):
        test_cases = _multilabel_test_cases()

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
                        msg=f"{strategy.__name__}'s multilabel test "
                        f"parameters contain no `SklearnClassifier` whose "
                        f"`proba_format` can be varied.",
                    )

    def test_multilabel_inventory_matches_probability_consumption(self):
        # A malformed output count must surface as a clear validation error
        # for every inventoried probability consumer, and must stay unnoticed
        # by every strategy inventoried as not consuming probabilities. This
        # keeps a misfiled strategy from passing the inventory silently.
        test_cases = _multilabel_test_cases()
        non_consumers = (
            MULTILABEL_PREDICTION_CONSUMERS | MULTILABEL_TASK_AGNOSTIC
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
