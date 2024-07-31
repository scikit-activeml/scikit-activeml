import inspect
import json
import os
import shutil
import unittest
from os import path

from docs.generate import generate_examples
from skactiveml import pool, stream

from skactiveml.stream import CognitiveDualQueryStrategy

QUERY_STRATEGY_EXCEPTIONS_LIST = [
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


class Dummy:
    def __init__(self):
        pass
