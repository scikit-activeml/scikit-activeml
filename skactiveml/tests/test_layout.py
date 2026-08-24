"""Enforces where shared helpers live.

`skactiveml/base.py` holds base classes. It may keep a private function or
constant serving the classes declared beside it, but such a helper must not
become a shared utility of the package: a module needing one imports it from
the subpackage owning it, or from a clearly named private module under
`skactiveml/utils`. `docs/contributing.rst` states the rule in prose; this
test is what keeps it true.

The check is static. It reads `base.py` to learn which of its module-level
names are classes, and reads every other module to learn which of those names
it refers to. Nothing is imported, so a class declared behind an optional
dependency counts as a class whether or not that dependency is installed.
"""

import ast
import unittest
from pathlib import Path

import skactiveml

_PACKAGE_ROOT = Path(skactiveml.__file__).parent
_BASE_PATH = _PACKAGE_ROOT / "base.py"
_BASE_MODULE = "skactiveml.base"


def _module_level_statements(body):
    """Yield the statements a module executes at import time.

    Conditional and guarded blocks are entered, because a class declared
    behind an optional dependency is still declared at module level. Class
    and function bodies are not, so that a method never shadows the name of
    a module-level declaration.
    """
    for node in body:
        if isinstance(node, (ast.If, ast.Try)):
            nested = [
                *node.body,
                *getattr(node, "orelse", []),
                *getattr(node, "finalbody", []),
            ]
            for handler in getattr(node, "handlers", []):
                nested.extend(handler.body)
            yield from _module_level_statements(nested)
        else:
            yield node


def _base_classes():
    """Collect the names `base.py` declares with a `class` statement."""
    tree = ast.parse(_BASE_PATH.read_text())
    return {
        node.name
        for node in _module_level_statements(tree.body)
        if isinstance(node, ast.ClassDef)
    }


def _containing_package(path):
    """Return the dotted package a module file belongs to."""
    parts = list(path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts)
    return ".".join(parts[:-1])


def _absolute_module(node, package):
    """Resolve the module a `from ... import ...` statement names.

    A relative import is resolved against the package containing the file, so
    that `..base` in `skactiveml/pool` and `skactiveml.base` are recognized as
    the same module. Matching the last component alone would also flag
    `sklearn.base`, which several modules import legitimately.
    """
    if not node.level:
        return node.module
    parts = package.split(".")
    if node.level > len(parts):
        return None
    prefix = parts[: len(parts) - node.level + 1]
    return ".".join([*prefix, node.module] if node.module else prefix)


def _dotted_name(node):
    """Return the dotted name of an attribute chain, or `None`."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _base_names_used(path):
    """Collect the `base.py` names one module refers to.

    Both spellings are covered: importing a name directly, and binding the
    module and reading a name off it.
    """
    tree = ast.parse(path.read_text())
    package = _containing_package(path)
    names = set()
    module_aliases = {_BASE_MODULE}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _absolute_module(node, package)
            if resolved == _BASE_MODULE:
                names.update(alias.name for alias in node.names)
            elif resolved == "skactiveml":
                module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "base"
                )
        elif isinstance(node, ast.Import):
            module_aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == _BASE_MODULE and alias.asname
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted_name(node)
        if dotted is None or "." not in dotted:
            continue
        prefix, _, attribute = dotted.rpartition(".")
        if prefix in module_aliases:
            names.add(attribute)
    return names


class TestLayout(unittest.TestCase):
    maxDiff = None

    def test_modules_import_only_classes_from_base(self):
        classes = _base_classes()
        self.assertIn(
            "SkactivemlClassifier",
            classes,
            msg="`base.py` could not be read for its class declarations.",
        )

        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            if path == _BASE_PATH:
                continue
            for name in sorted(_base_names_used(path)):
                if name not in classes:
                    relative = path.relative_to(_PACKAGE_ROOT.parent)
                    offenders.append(f"{relative}: {name}")

        self.assertEqual(
            [],
            offenders,
            msg="`skactiveml/base.py` is for base classes. Keep a helper "
            "function or constant in the file that uses it, in a private "
            "file of the subpackage sharing it, or in a clearly named "
            "private file under `skactiveml/utils`. See the helper "
            "placement rule in `docs/contributing.rst`.",
        )
