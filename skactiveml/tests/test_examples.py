import inspect
import json
import os
import shutil
import tempfile
import unittest
from os import path

from docutils.utils import column_width

from docs.generate import (
    OVERVIEW_HEADINGS,
    generate_examples,
    generate_strategy_overview_rst,
)
from skactiveml import pool, stream

from skactiveml.pool import ExpectedErrorReduction
from skactiveml.pool.tests.test_multilabel_contracts import (
    MULTILABEL_DELEGATING_WRAPPERS,
    MULTILABEL_PREDICTION_CONSUMERS,
    MULTILABEL_PROBA_CONSUMERS,
    MULTILABEL_TASK_AGNOSTIC,
)
from skactiveml.stream import UncertaintyZliobaite, CognitiveDualQueryStrategy

QUERY_STRATEGY_EXCEPTIONS_LIST = [
    ExpectedErrorReduction,
    UncertaintyZliobaite,
    CognitiveDualQueryStrategy,
]


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.skaml_path = path.abspath(os.curdir).split("skactiveml")[0]
        self.docs_path = path.join(self.skaml_path, "docs")
        self.json_path = path.join(self.skaml_path, "docs", "examples")
        self.exceptions = [
            qs.__name__ for qs in QUERY_STRATEGY_EXCEPTIONS_LIST
        ]
        self.working_dir = os.path.abspath(os.curdir)

        # A list of all modules that should have a json file.
        self.modules = [pool, stream]

    def test_example_files(self):
        # Temporary generate the examples from the json files.
        examples_path = path.join(self.skaml_path, "docs", "temp_examples")
        notebooks_path = path.join(self.skaml_path, "docs", "temp_notebooks")
        os.chdir(self.docs_path)
        generate_examples(examples_path, self.json_path, notebooks_path)
        os.chdir(self.working_dir)

        # Execute the examples.
        for root, dirs, files in os.walk(examples_path, topdown=True):
            for filename in files:
                if filename.endswith(".py"):
                    msg = os.path.join(root, filename).replace(
                        examples_path, ""
                    )
                    file_path = path.join(root, filename)
                    with self.subTest(msg=msg):
                        with open(file_path, "r") as f:
                            exec(f.read(), locals())

        # Remove the created examples and notebooks from disk.
        shutil.rmtree(examples_path)

    def test_json(self):
        # Collect all strategies for which an example exists
        strats_with_json = []
        for root, dirs, files in os.walk(self.json_path, topdown=True):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                with open(path.join(root, filename)) as file:
                    for example in json.load(file):
                        if example["class"] not in strats_with_json:
                            strats_with_json.append(example["class"])

        # Test if there is a json example for every AL-strategy.
        for module in self.modules:
            for item in module.__all__:
                with self.subTest(msg="JSON Test", qs_name=item):
                    item_missing = (
                        inspect.isclass(getattr(module, item))
                        and item not in self.exceptions
                        and item not in strats_with_json
                    )
                    self.assertFalse(
                        item_missing,
                        f'No json example found for "{item}". Please '
                        f"add an example in\n"
                        f"{self.json_path}.\n"
                        f"For information how to create one, see the "
                        f"Developers Guide. If {item} is not an "
                        f'AL-strategy, add "{item}" to the '
                        f'"exceptions" list in this test class.',
                    )

    def test_multilabel_tags_match_capability_inventory(self):
        expected_strategies = {
            strategy.__name__
            for strategy in (
                MULTILABEL_PROBA_CONSUMERS
                | MULTILABEL_PREDICTION_CONSUMERS
                | MULTILABEL_TASK_AGNOSTIC
                | MULTILABEL_DELEGATING_WRAPPERS
            )
        }
        examples_by_strategy = {}
        for root, dirs, files in os.walk(self.json_path, topdown=True):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                with open(path.join(root, filename)) as file:
                    for example in json.load(file):
                        examples_by_strategy.setdefault(
                            example["class"], []
                        ).append(example)

        tagged_strategies = {
            strategy
            for strategy, examples in examples_by_strategy.items()
            if any("multi-label" in example["tags"] for example in examples)
        }
        # The tags are not rendered; they only decide which rows the
        # `Multi-Label` filter of the strategy overview surfaces. What a user
        # depends on is therefore that no multi-label capable strategy is
        # missing from that filter, which is what this comparison states.
        # Whether every single example of such a strategy is tagged is not
        # checked, because a class declares its capabilities while an example
        # describes one configuration of it, e.g.
        # `UncertaintySampling(method="expected_average_precision")` is not
        # multi-label capable although its class is.
        self.assertEqual(expected_strategies, tagged_strategies)

    @staticmethod
    def _overview_example(qs_name, method, category):
        return {
            "class": qs_name,
            "package": "pool",
            "method": method,
            "category": category,
            "tags": ["pool", "classification", "single-annotator"],
            "refs": [],
        }

    def _generate_overview(self, json_data, sections=()):
        """Render an overview and return it, with a stub for each section."""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        gen_path = path.join(tmp_dir, "generated")
        os.makedirs(gen_path)
        for section in sections:
            section_path = path.join(tmp_dir, "examples", section)
            os.makedirs(section_path)
            with open(path.join(section_path, "README.rst"), "w") as file:
                file.write(f"{section}\n")
        generate_strategy_overview_rst(gen_path, json_data)
        with open(path.join(gen_path, "strategy_overview.rst")) as file:
            return file.read()

    def test_strategy_overview_groups_tasks_under_their_scenario(self):
        # Two sections of one scenario share its heading, and each task and
        # category is one level below the previous one.
        sections = list(OVERVIEW_HEADINGS)[:2]
        scenario, first_task = OVERVIEW_HEADINGS[sections[0]]
        _, second_task = OVERVIEW_HEADINGS[sections[1]]
        json_data = {
            sections[0]: {
                "data": [
                    self._overview_example(
                        "RandomSampling", "Random Sampling", "Baseline"
                    )
                ]
            },
            sections[1]: {
                "data": [
                    self._overview_example(
                        "GreedySamplingX", "Greedy Sampling", "Informativeness"
                    )
                ]
            },
        }

        overview = self._generate_overview(json_data, sections)

        self.assertEqual(
            overview.count(f"{scenario}\n"),
            1,
            msg="One scenario heading must cover all of its tasks.",
        )
        for title, underline in [
            (scenario, "-"),
            (first_task, "~"),
            (second_task, "~"),
            ("Baseline", "^"),
            ("Informativeness", "^"),
        ]:
            with self.subTest(title=title):
                # `docutils` measures an underline by its display width, so
                # a title containing an emoji needs more characters than it
                # has.
                expected = "".ljust(column_width(title), underline)
                self.assertIn(f"{title}\n{expected}\n", overview)

    def test_strategy_overview_keeps_a_section_without_a_heading_path(self):
        # A gallery section that nobody added to `OVERVIEW_HEADINGS` keeps
        # its own title instead of silently losing its strategies.
        json_data = {
            "9-unmapped": {
                "data": [
                    self._overview_example(
                        "RandomSampling", "Random Sampling", "Baseline"
                    )
                ]
            }
        }

        with self.assertWarnsRegex(UserWarning, "OVERVIEW_HEADINGS"):
            overview = self._generate_overview(json_data, ["9-unmapped"])

        self.assertIn("9-unmapped\n", overview)
        self.assertIn("RandomSampling", overview)

    def test_strategy_overview_offers_a_filter_per_tag(self):
        # The tags themselves are never rendered; a strategy reaches a user
        # only through these checkboxes, so every tag a row can carry needs
        # one. The rows are real, because a filter over nothing filters
        # nothing.
        section = next(iter(OVERVIEW_HEADINGS))
        json_data = {
            section: {
                "data": [
                    self._overview_example(
                        "RandomSampling", "Random Sampling", "Baseline"
                    )
                ]
            }
        }

        overview = self._generate_overview(json_data, [section])

        for tag in [
            "classification",
            "regression",
            "multi-label",
            "single-annotator",
            "multi-annotator",
            "top-k-batch",
            "diverse-batch",
        ]:
            with self.subTest(tag=tag):
                self.assertIn(f'value="{tag}"', overview)
        self.assertIn("<label>Multi-Label</label>", overview)
        # The filter reads the tags of a rendered row, so the row has to
        # carry them where the filter looks.
        self.assertIn("RandomSampling", overview)
        self.assertIn("single-annotator", overview)


class Dummy:
    def __init__(self):
        pass
