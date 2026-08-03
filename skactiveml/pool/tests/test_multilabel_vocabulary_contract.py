"""Inventory of the custom class-vocabulary contract of pool strategies.

A multi-label target may use its own binary class vocabulary per label output,
including string labels such as `[["no", "yes"], ["off", "on"]]`. Pool
strategies must therefore operate on encoded targets instead of assuming that
raw labels are numeric or that they are `{0, 1}`.

Every strategy supplying `query_default_params_clf_multilabel` is covered by
the shared contract test `test_query_multilabel_custom_class_vocabularies` of
`TemplatePoolQueryStrategy`, which runs a successful query with distinct
string vocabularies per output and compares it against the semantically
equivalent numeric encoding.

Most test cases use the default fixture, which relabels the target of their
default multi-label query parameters. A strategy whose query needs auxiliary
inputs that the default fixture cannot derive, e.g. embeddings, logits, or a
discriminator, provides its own fixture by overriding
`_multilabel_custom_vocabulary_params`. The tests below record both groups, so
that neither a newly added multi-label strategy nor a newly added fixture
override can skip the contract unnoticed.
"""

import unittest

import numpy as np

from skactiveml.pool.tests._multilabel_target_semantics import (
    collect_pool_template_test_cases,
)
from skactiveml.tests.template_query_strategy import (
    TemplatePoolQueryStrategy,
)

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

# Test cases providing their own fixture, with the reason why the default one
# cannot cover the acquisition path in question.
SPECIALIZED_VOCABULARY_FIXTURE = {
    "TestUHerding": "needs logits and a multi-element temperature grid to "
    "reach per-output calibration",
}

# The shared contract tests of `TemplatePoolQueryStrategy` that no test case
# may replace, because doing so would silently drop the contract.
CONTRACT_TEST_NAMES = (
    "test_query_multilabel_custom_class_vocabularies",
    "test_query_multilabel_invalid_rows",
)


def _multilabel_test_cases():
    """Collects the pool test cases defining multi-label query parameters."""
    return [
        test_case
        for test_case in collect_pool_template_test_cases()
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
        # The contract requires one distinct vocabulary per label output, each
        # ordered such that its second entry is the positive class, i.e. the
        # class whose probability column the strategies consume.
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
                for test_case in _multilabel_test_cases()
            },
            inventory,
            msg="A multi-label pool strategy test is missing from the "
            "custom class-vocabulary inventory. Record it as using the "
            "default fixture or as providing a specialized override.",
        )

    def test_inventory_matches_the_used_fixture(self):
        for test_case in _multilabel_test_cases():
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
        for test_case in _multilabel_test_cases():
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
