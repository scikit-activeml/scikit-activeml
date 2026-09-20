import inspect
import json
import os
import shutil
import tempfile
import unittest
from os import path

from docutils.utils import column_width
from jinja2 import Environment, FileSystemLoader

from docs.generate import (
    OVERVIEW_SCENARIOS,
    generate_examples,
    generate_strategy_overview_rst,
    get_skactiveml_methods,
)
from skactiveml import pool, stream
from skactiveml.base import QueryStrategy

from skactiveml.pool import ExpectedErrorReduction, multiannotator
from skactiveml.pool.tests._strategy_helpers import _instantiate
from skactiveml.pool.tests.test_multilabel_contracts import (
    MULTILABEL_CAPABILITY,
)
from skactiveml.stream import UncertaintyZliobaite, CognitiveDualQueryStrategy

QUERY_STRATEGY_EXCEPTIONS_LIST = [
    ExpectedErrorReduction,
    UncertaintyZliobaite,
    CognitiveDualQueryStrategy,
]

EXAMPLE_SUBPACKAGES = (pool, multiannotator, stream)


def _multilabel_capable_strategy_names():
    """Collect the names of the strategies declaring multi-label support."""
    names = set()
    for module in EXAMPLE_SUBPACKAGES:
        for name in dir(module):
            obj = getattr(module, name)
            if not inspect.isclass(obj) or not issubclass(obj, QueryStrategy):
                continue
            capabilities = _instantiate(obj)._target_capabilities
            if MULTILABEL_CAPABILITY in capabilities:
                names.add(obj.__name__)
    return names


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

    def test_api_reference_prioritizes_native_methods(self):
        environment = Environment(loader=FileSystemLoader(self.docs_path))
        environment.filters["underline"] = lambda value: value
        template = environment.get_template("_templates/class.rst")
        skactiveml_methods = get_skactiveml_methods()
        test_cases = [
            (
                "skactiveml.classifier.SklearnClassifier",
                [
                    "fit",
                    "get_metadata_routing",
                    "get_params",
                    "predict",
                    "score",
                    "set_fit_request",
                    "set_params",
                ],
                [
                    "fit",
                    "predict",
                    "score",
                    "get_metadata_routing",
                    "get_params",
                    "set_fit_request",
                    "set_params",
                ],
            ),
            (
                "skactiveml.pool.UncertaintySampling",
                ["get_params", "query", "set_params"],
                ["query", "get_params", "set_params"],
            ),
        ]
        for class_name, methods, expected in test_cases:
            with self.subTest(class_name=class_name):
                module, name = class_name.rsplit(".", 1)
                rendered = template.render(
                    objname=name,
                    module=module,
                    fullname=class_name,
                    name=name,
                    methods=methods,
                    attributes=["attribute"],
                    skactiveml_methods=skactiveml_methods,
                    _=lambda value: value,
                )
                summarized = [
                    line.strip().rsplit(".", 1)[-1]
                    for line in rendered.splitlines()
                    if line.strip().startswith(f"~{name}.")
                    and line.strip().rsplit(".", 1)[-1] in methods
                ]
                detailed = [
                    line.strip().rsplit(".", 1)[-1]
                    for line in rendered.splitlines()
                    if line.strip().startswith(".. automethod::")
                ]
                self.assertIn(":no-members:", rendered)
                self.assertIn(":no-inherited-members:", rendered)
                self.assertIn(
                    f".. autoattribute:: {class_name}.attribute", rendered
                )
                self.assertEqual(expected, summarized)
                self.assertEqual(expected, detailed)

    def test_multilabel_tags_match_declared_capabilities(self):
        expected_strategies = _multilabel_capable_strategy_names()
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
        # Each class supporting multi-label targets needs a tagged example.
        # Individual configurations may be unsupported, such as
        # `UncertaintySampling(method="expected_average_precision")`.
        self.assertEqual(expected_strategies, tagged_strategies)

    @staticmethod
    def _overview_example(
        qs_name, method, category, package="pool", tags=None
    ):
        return {
            "class": qs_name,
            "package": package,
            "method": method,
            "category": category,
            "tags": (
                [package.split(".")[0], "classification", "single-annotator"]
                if tags is None
                else tags
            ),
            "refs": [],
        }

    def _generate_overview(self, json_data):
        """Render an overview independently of gallery section headings."""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        gen_path = path.join(tmp_dir, "generated")
        os.makedirs(gen_path)
        generate_strategy_overview_rst(gen_path, json_data)
        with open(path.join(gen_path, "strategy_overview.rst")) as file:
            return file.read()

    def test_strategy_overview_has_one_table_per_scenario(self):
        # Merge tasks and annotator settings while retaining every example
        # and its original link, including methods that support both tasks.
        json_data = {
            "1-pool-classification": {
                "data": [
                    self._overview_example(
                        "RandomSampling", "Random Sampling", "Baseline"
                    )
                ]
            },
            "2-pool-regression": {
                "data": [
                    self._overview_example(
                        "GreedySamplingX",
                        "Greedy Sampling",
                        "Representativeness",
                        tags=[
                            "pool",
                            "classification",
                            "regression",
                            "single-annotator",
                        ],
                    )
                ]
            },
            "3-pool-multi-annotator": {
                "data": [
                    self._overview_example(
                        "SingleAnnotatorWrapper",
                        "Random Sampling",
                        "Baseline",
                        package="pool.multiannotator",
                        tags=["pool", "classification", "multi-annotator"],
                    )
                ]
            },
            "4-stream": {
                "data": [
                    self._overview_example(
                        "StreamRandomSampling",
                        "Stream Random Sampling",
                        "Baseline",
                        package="stream",
                    )
                ]
            },
        }

        overview = self._generate_overview(json_data)

        self.assertEqual(overview.count(".. list-table::"), 2)
        for title in OVERVIEW_SCENARIOS.values():
            with self.subTest(title=title):
                self.assertEqual(overview.count(f"{title}\n"), 1)
                self.assertIn(
                    f"{title}\n{'-' * column_width(title)}\n", overview
                )
        pool_table, stream_table = overview.split(
            OVERVIEW_SCENARIOS["pool"] + "\n", 1
        )[1].split(OVERVIEW_SCENARIOS["stream"] + "\n", 1)
        self.assertEqual(pool_table.count(".. list-table::"), 1)
        self.assertEqual(stream_table.count(".. list-table::"), 1)
        self.assertEqual(
            pool_table.count("</generated/sphinx_gallery_examples/"), 3
        )
        self.assertEqual(
            stream_table.count("</generated/sphinx_gallery_examples/"), 1
        )
        self.assertIn(
            "3-pool-multi-annotator/plot-SingleAnnotatorWrapper-", pool_table
        )
        self.assertLess(
            pool_table.index(":doc:`Greedy Sampling"),
            pool_table.index(":doc:`Random Sampling"),
        )
        self.assertIn("| Classification, Regression", pool_table)
        self.assertIn("| Multiple annotators", pool_table)
        self.assertIn("| **Representativeness**", pool_table)
        self.assertIn("API class", pool_table)
        self.assertNotIn("Base Class", overview)
        self.assertNotIn("\nClassification\n", overview)
        self.assertNotIn("\nRegression\n", overview)
        self.assertNotIn("\nBaseline\n", overview)

    def test_strategy_overview_keeps_methods_from_new_gallery_sections(self):
        # A new gallery section needs no overview heading configuration.
        json_data = {
            "9-unmapped": {
                "data": [
                    self._overview_example(
                        "RandomSampling", "Random Sampling", "Baseline"
                    )
                ]
            }
        }

        overview = self._generate_overview(json_data)

        self.assertIn("Pool-based Active Learning\n", overview)
        self.assertEqual(overview.count(".. list-table::"), 1)
        self.assertIn(
            "9-unmapped/plot-RandomSampling-Random_Sampling", overview
        )
        self.assertNotIn("\n9-unmapped\n", overview)

    def test_strategy_overview_offers_a_filter_per_tag(self):
        section = "1-pool-classification"
        json_data = {
            section: {
                "data": [
                    self._overview_example(
                        "RandomSampling", "Random Sampling", "Baseline"
                    )
                ]
            }
        }

        overview = self._generate_overview(json_data)

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
        self.assertIn("<span>Multi-label</span>", overview)
        # The filter reads the tags of a rendered row, so the row has to
        # carry them where the filter looks.
        self.assertIn("RandomSampling", overview)
        self.assertIn("single-annotator", overview)


class Dummy:
    def __init__(self):
        pass
