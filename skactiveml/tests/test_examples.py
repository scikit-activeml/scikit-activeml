import inspect
import json
import os
import re
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
    generate_visualizations_rst,
    generate_documentation_readme,
    post_process_tutorials,
    get_strategy_metadata,
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
        templates_path = path.join(self.docs_path, "_templates")
        environment = Environment(
            loader=FileSystemLoader([self.docs_path, templates_path])
        )
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
                + (["random-batch"] if package != "stream" else [])
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
        # Merge tasks and annotator settings, linking to complete API galleries
        # instead of selecting one demonstration for each method.
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
                            "diverse-batch",
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
                        tags=[
                            "pool",
                            "classification",
                            "multi-annotator",
                            "random-batch",
                        ],
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
        self.assertNotIn("/plot-", overview)
        self.assertNotIn("*Example:", overview)
        self.assertEqual(pool_table.count("</generated/api/"), 3)
        self.assertEqual(stream_table.count("</generated/api/"), 1)
        self.assertLess(
            pool_table.index("Greedy Sampling"),
            pool_table.index("Random Sampling"),
        )
        self.assertIn("| Single-output classification", pool_table)
        self.assertIn("| Regression", pool_table)
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
        self.assertIn("Random Sampling", overview)
        self.assertIn(
            "/generated/api/skactiveml.pool.RandomSampling", overview
        )
        self.assertNotIn("9-unmapped", overview)
        self.assertNotIn("\n9-unmapped\n", overview)

    def test_strategy_overview_offers_one_choice_per_setup_property(self):
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
            "single-output",
            "multi-label",
            "single-annotator",
            "multi-annotator",
            "top-k-batch",
            "diverse-batch",
            "random-batch",
            "single-sample",
            "batch",
        ]:
            with self.subTest(tag=tag):
                self.assertIn(f'value="{tag}"', overview)
        self.assertIn("<span>Multi-label</span>", overview)
        self.assertEqual(overview.count('value="" checked'), 4)
        self.assertNotIn('type="checkbox"', overview)
        for name in ["task", "target", "annotators", "batch"]:
            self.assertIn(f'name="strategy-{name}"', overview)
        self.assertIn("RandomSampling", overview)
        self.assertIn(
            '"capabilities": [["classification", "single-output", '
            '"single-annotator"]]',
            overview,
        )

    def test_strategy_overview_metadata_for_documented_variants(self):
        records = {}
        for root, _, files in os.walk(self.json_path):
            for filename in files:
                if filename.endswith(".json"):
                    with open(path.join(root, filename)) as file:
                        for data in json.load(file):
                            records[(data["class"], data["method"])] = (
                                get_strategy_metadata(data)
                            )

        def metadata(cls, method=None):
            matches = [
                value
                for (name, label), value in records.items()
                if name == cls and (method is None or label == method)
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        # Combining supported properties must not invent multilabel regression.
        core_set = metadata("CoreSet")
        self.assertIn(
            ("regression", "single-output", "single-annotator"),
            core_set["capabilities"],
        )
        self.assertIn(
            ("classification", "multi-label", "single-annotator"),
            core_set["capabilities"],
        )
        for record in records.values():
            self.assertNotIn(
                ("regression", "multi-label", "single-annotator"),
                record["capabilities"],
            )

        # Native multilabel methods cannot be offered for single-output use.
        for name in [
            "LabelCardinalityInconsistency",
            "MaxLossReductionMaxConfidence",
        ]:
            self.assertEqual(
                metadata(name)["capabilities"],
                [("classification", "multi-label", "single-annotator")],
            )
        usap = metadata(
            "UncertaintySampling",
            "Uncertainty Sampling with Expected Average Precision (USAP)",
        )
        self.assertEqual(
            usap["capabilities"],
            [("classification", "single-output", "single-annotator")],
        )

        # Pool batches, single-only wrappers, and stream budgets differ.
        qbc = metadata(
            "QueryByCommittee",
            "Query-by-Committee (QBC) with Empirical Variance",
        )
        self.assertEqual(qbc["batch"], ["top-k-batch"])
        self.assertEqual(
            qbc["capabilities"],
            [("regression", "single-output", "single-annotator")],
        )
        self.assertEqual(
            metadata("ParallelUtilityEstimationWrapper")["batch"],
            ["single-sample"],
        )
        self.assertEqual(metadata("RandomSampling")["batch"], ["random-batch"])
        self.assertEqual(metadata("StreamRandomSampling")["batch"], [])
        self.assertEqual(metadata("PeriodicSampling")["batch"], [])
        self.assertEqual(
            metadata("IntervalEstimationThreshold")["batch"], ["top-k-batch"]
        )
        self.assertEqual(
            metadata("SingleAnnotatorWrapper", "Core Set + Greedy Selection")[
                "batch"
            ],
            ["diverse-batch"],
        )
        for name in ["SubSamplingWrapper", "ParallelUtilityEstimationWrapper"]:
            self.assertTrue(metadata(name)["conditional"])

    def test_strategy_overview_rejects_invalid_metadata(self):
        data = self._overview_example(
            "LabelCardinalityInconsistency", "LCI", "Informativeness"
        )
        with self.assertRaisesRegex(ValueError, "capability tags"):
            get_strategy_metadata(data)
        data = self._overview_example(
            "StreamRandomSampling", "Random", "Baseline", package="stream"
        )
        data["tags"].append("top-k-batch")
        with self.assertRaisesRegex(ValueError, "pool batch tags"):
            get_strategy_metadata(data)
        data = self._overview_example("RandomSampling", "Random", "Baseline")
        data["tags"].append("single-sample")
        with self.assertRaisesRegex(ValueError, "pool batching"):
            get_strategy_metadata(data)

    def test_visualizations_group_by_demonstrated_setup(self):
        json_data = {}
        for root, _, files in os.walk(self.json_path):
            records = []
            for filename in sorted(files):
                if filename.endswith(".json"):
                    with open(path.join(root, filename)) as file:
                        records.extend(json.load(file))
            if records:
                json_data[path.relpath(root, self.json_path)] = {
                    "data": records
                }

        with tempfile.TemporaryDirectory() as directory:
            generate_visualizations_rst(
                directory, json_data, examples_path="generated/examples"
            )
            with open(path.join(directory, "index.rst")) as file:
                index = file.read()
            section_indexes = {}
            for section in json_data:
                with open(path.join(directory, section, "index.rst")) as file:
                    section_indexes[section] = file.read()

        all_examples = re.findall(
            r"^\s+(generated/examples/.*\.py)$", index, re.MULTILINE
        )
        expected_count = sum(
            len(section["data"]) for section in json_data.values()
        )
        self.assertEqual(len(all_examples), expected_count)
        self.assertEqual(len(set(all_examples)), expected_count)

        def illustrated_examples(section):
            return re.findall(
                r"^\s+(generated/examples/.*\.py)$",
                section_indexes[section],
                re.MULTILINE,
            )

        classification = illustrated_examples("1-pool-classification")
        regression = illustrated_examples("2-pool-regression")
        stream = illustrated_examples("4-stream")
        # Broad capabilities must not duplicate a classification animation
        # under regression or multilabel. Keep the existing example URL.
        random = (
            "generated/examples/1-pool-classification/"
            "plot-RandomSampling-Random_Sampling.py"
        )
        self.assertIn(random, classification)
        self.assertNotIn(random, regression + stream)
        for cls in [
            "LabelCardinalityInconsistency",
            "MaxLossReductionMaxConfidence",
        ]:
            self.assertTrue(
                any(f"/plot-{cls}-" in item for item in classification)
            )
        self.assertTrue(
            all("/2-pool-regression/" in item for item in regression)
        )
        self.assertEqual(
            sum("/3-pool-multi-annotator/" in item for item in classification),
            3,
        )
        self.assertTrue(all("/4-stream/" in item for item in stream))
        self.assertIn(
            "Strategy Overview </generated/strategy_overview>", index
        )
        # Each example has exactly one task-page owner, independent of its
        # physical source directory. The legacy annotation collection remains.
        owned_documents = []
        for section in [
            "1-pool-classification",
            "2-pool-regression",
            "4-stream",
        ]:
            self.assertIn(f"   {section}/index", index)
            owned_documents.extend(
                re.findall(
                    r"^\s+(/generated/sphinx_gallery_examples/.*)$",
                    section_indexes[section],
                    re.MULTILINE,
                )
            )
        self.assertEqual(len(owned_documents), expected_count)
        self.assertEqual(len(set(owned_documents)), expected_count)
        legacy = section_indexes["3-pool-multi-annotator"]
        self.assertIn(":orphan:", legacy)
        self.assertNotIn(".. toctree::", legacy)
        self.assertNotIn("   3-pool-multi-annotator/index", index)
        self.assertIn("Diverse · configured query size: 4 samples", index)
        self.assertIn("configured query size: 3 sample–annotator pairs", index)
        self.assertNotIn("configured query size:", section_indexes["4-stream"])

    def test_visualizations_reject_unhandled_setups(self):
        data = self._overview_example(
            "StreamRandomSampling", "Random", "Baseline", package="stream"
        )
        data["template"] = "examples/template_stream_regression.py"
        data["example_setup"] = {
            "task": "regression",
            "target": "single-output",
            "annotators": "single-annotator",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No visualization group"):
                generate_visualizations_rst(
                    directory,
                    {"4-stream": {"data": [data]}},
                    examples_path="generated/examples",
                )

    def test_documentation_readme_links_keep_build_version(self):
        readme = path.join(self.skaml_path, "README.rst")
        with open(readme) as file:
            original = file.read()
        with tempfile.TemporaryDirectory() as directory:
            destination = path.join(directory, "readme.rst")
            generate_documentation_readme(readme, destination)
            with open(destination) as file:
                rendered = file.read()
        self.assertNotIn("scikit-activeml.github.io/latest/", rendered)
        self.assertIn("Strategy Overview", rendered)
        self.assertIn("</generated/strategy_overview>`", rendered)
        self.assertIn(
            "</generated/tutorials/30_pool_multilabel_getting_started>`",
            rendered,
        )
        self.assertIn(
            ".. image:: /logos/scikit-activeml-query-strategy-overview.svg",
            rendered,
        )
        self.assertIn(".. image:: /logos/readme_pool.gif", rendered)
        self.assertIn(".. image:: /logos/readme_stream.gif", rendered)
        with open(readme) as file:
            self.assertEqual(file.read(), original)

    def test_tutorial_documentation_links_in_colab_match_version(self):
        for version in ["latest", "development", "1.0"]:
            for colab in [False, True]:
                with self.subTest(version=version, colab=colab):
                    notebook = {
                        "metadata": {},
                        "cells": [
                            {
                                "cell_type": "markdown",
                                "source": [
                                    "[Targets](../../target_semantics.rst"
                                    "#classification)\n",
                                    "[Strategies](../strategy_overview.rst)\n",
                                    "[Notebook]"
                                    "(00_pool_getting_started.ipynb)",
                                ],
                            },
                        ],
                    }
                    with tempfile.TemporaryDirectory() as directory:
                        filename = path.join(directory, "example.ipynb")
                        with open(filename, "w") as file:
                            json.dump(notebook, file)
                        post_process_tutorials(
                            directory,
                            "generated/tutorials_colab",
                            show_installation_code=colab,
                            version=version,
                        )
                        with open(filename) as file:
                            source = "".join(
                                json.load(file)["cells"][0]["source"]
                            )
                    prefix = f"https://scikit-activeml.github.io/{version}/"
                    targets = (
                        prefix + "target_semantics.html#classification"
                        if colab
                        else "../../target_semantics.rst#classification"
                    )
                    strategies = (
                        prefix + "generated/strategy_overview.html"
                        if colab
                        else "../strategy_overview.rst"
                    )
                    self.assertIn(f"[Targets]({targets})", source)
                    self.assertIn(f"[Strategies]({strategies})", source)
                    self.assertIn(
                        "[Notebook](00_pool_getting_started.ipynb)", source
                    )


class Dummy:
    def __init__(self):
        pass
