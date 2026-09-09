"""The label and missing-value contract, component by component.

Every case below mirrors one row of the tables in the contract section of
`docs/target_semantics.rst`, in their order, and asserts that every component
consuming labels reaches the same verdict for it. A rule that only held for
some containers or some components would not be part of the contract, so each
row runs for a Python list, the equivalent NumPy array, and an object array,
and against every component the row's verdict can be reached by.

A row is scoped by the evidence its verdict needs. `LABELS` rows follow from
`y` and `missing_label` alone, so even a component without a class vocabulary
must reach them. `CLASSES` rows need a class vocabulary, so only the
components resolving one are asked; a strategy such as `RandomSampling`
never learns which classes exist and therefore has nothing to reject.

`TestDocumentedOutcomeTables` locks that correspondence in the other
direction: a row added to one of the outcome tables without a case here, or
a case dropped from here, fails.

The tables say which inputs are accepted and which are rejected. The closing
`TestClassIdentityRoundTrip` covers the other half of the promise: that an
accepted class label comes back as the value it was declared as. Asserting
that needs an estimator whose probabilities are analytically known rather
than a sweep over components, so it names its vehicle instead of hiding it.
"""

import re
import unittest
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import RandomSampling, UncertaintySampling
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.utils import (
    ExtLabelEncoder,
    is_unlabeled,
    majority_vote,
    resolve_target_spec,
)

# The evidence a row's verdict needs.
LABELS = "labels"
CLASSES = "classes"

# Every row is repeated so that estimators see enough samples per class to
# be fitted; the values a row is about stay the same.
REPEATS = 3

Row = namedtuple(
    "Row",
    [
        "classes",
        "missing_label",
        "y",
        "error",
        "message",
        "scope",
        "skip",
        "foreign_message",
    ],
)
Row.__new__.__defaults__ = (LABELS, (), ())

RegressionRow = namedtuple(
    "RegressionRow", ["missing_label", "y", "error", "message", "skip"]
)
RegressionRow.__new__.__defaults__ = ((),)


def _containers(y):
    """Return the row's labels in each container the contract covers."""
    values = list(y) * REPEATS
    return (
        ("list", values),
        ("array", np.asarray(values)),
        ("object", np.asarray(values, dtype=object)),
    )


def _multilabel_containers(rows):
    """Return the row's label matrix in each container."""
    values = [list(row) for row in rows] * REPEATS
    return (
        ("list", values),
        ("array", np.asarray(values)),
        ("object", np.asarray(values, dtype=object)),
    )


def _samples(y):
    """Return well-separated samples for as many labels as `y` holds."""
    return np.linspace(0.0, 1.0, len(y)).reshape(-1, 1)


def _annotator_column(y):
    """Return `y` as the single annotator column `majority_vote` expects."""
    if isinstance(y, np.ndarray):
        return y.reshape(-1, 1)
    return [[value] for value in y]


class _ContractCase(unittest.TestCase):
    """Assert one row against one component in every container."""

    def assert_row(self, name, row, run, containers):
        for container, y in containers:
            with self.subTest(component=name, container=container):
                X = _samples(y)
                if row.error is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        run(X, y)
                    continue
                with self.assertRaises(row.error) as caught:
                    run(X, y)
                if name not in row.foreign_message:
                    self.assertIn(row.message, str(caught.exception))


def _single_output_components(classes, missing_label):
    """Return every component consuming single-output classification labels."""

    def parzen_window_classifier(X, y):
        ParzenWindowClassifier(
            classes=classes, missing_label=missing_label
        ).fit(X, y)

    def sklearn_classifier(X, y):
        SklearnClassifier(
            GaussianNB(), classes=classes, missing_label=missing_label
        ).fit(X, y)

    def uncertainty_sampling(X, y):
        UncertaintySampling(missing_label=missing_label).query(
            X,
            y,
            clf=ParzenWindowClassifier(
                classes=classes, missing_label=missing_label
            ),
            candidates=X,
        )

    return {
        "is_unlabeled": (
            LABELS,
            lambda X, y: is_unlabeled(y, missing_label),
        ),
        "RandomSampling": (
            LABELS,
            lambda X, y: RandomSampling(missing_label=missing_label).query(
                X, y, candidates=X
            ),
        ),
        "resolve_target_spec": (
            CLASSES,
            lambda X, y: resolve_target_spec(
                y,
                task="classification",
                classes=classes,
                missing_label=missing_label,
            ),
        ),
        "ExtLabelEncoder": (
            CLASSES,
            lambda X, y: ExtLabelEncoder(
                classes=classes, missing_label=missing_label
            ).fit_transform(y),
        ),
        "majority_vote": (
            CLASSES,
            lambda X, y: majority_vote(
                _annotator_column(y),
                classes=classes,
                missing_label=missing_label,
            ),
        ),
        "ParzenWindowClassifier": (CLASSES, parzen_window_classifier),
        "SklearnClassifier": (CLASSES, sklearn_classifier),
        "UncertaintySampling": (CLASSES, uncertainty_sampling),
    }


class TestSingleOutputContract(_ContractCase):
    """The rows of the single-output classification outcome table."""

    ROWS = (
        Row([0, 1], np.nan, [0, np.nan, 1], None, ""),
        Row([0, 1], -2.5, [0, -2.5, 1], None, ""),
        Row([False, True], None, [False, None, True], None, ""),
        Row([0, 1], np.nan, [True, np.nan, False], None, ""),
        Row(["cat", "dog"], "?", ["cat", "?", "dog"], None, ""),
        Row(
            [0, 1],
            np.nan,
            [0, 2, np.nan],
            ValueError,
            "outside `classes`",
            scope=CLASSES,
            # The encoder rejects the undeclared class through the wrapped
            # `LabelEncoder`, which words it as an unseen label.
            foreign_message=("ExtLabelEncoder", "majority_vote"),
        ),
        Row(
            [0, 1],
            1,
            [0, 1],
            ValueError,
            "contains `missing_label=1`",
            scope=CLASSES,
        ),
        Row(
            [0, 1.5],
            np.nan,
            [0.0, 1.5],
            TypeError,
            "must contain one label family",
            scope=CLASSES,
        ),
        Row(
            None,
            None,
            [0.0, np.nan, 1.0],
            ValueError,
            "contains NaN",
        ),
        Row(
            None,
            "?",
            ["cat", "?", "dog"],
            None,
            "",
        ),
        Row(
            None,
            np.nan,
            [np.nan, np.nan],
            ValueError,
            "No class label is observed",
            scope=CLASSES,
            # Aggregation is the one consumer that needs no vocabulary for
            # such a matrix; the test below states what it returns instead.
            skip=("majority_vote",),
        ),
    )

    def test_rows(self):
        for row in self.ROWS:
            components = _single_output_components(
                row.classes, row.missing_label
            )
            for name, (evidence, run) in components.items():
                if name in row.skip:
                    continue
                expected = (
                    row
                    if row.scope == LABELS or evidence == CLASSES
                    else row._replace(error=None, message="")
                )
                self.assert_row(name, expected, run, _containers(row.y))

    def test_majority_vote_needs_no_vocabulary_for_a_missing_matrix(self):
        # The row rejecting an entirely missing `y` with `classes=None`
        # covers the consumers that have to infer a vocabulary from it.
        # Majority voting infers none, because every sample of such a matrix
        # keeps the missing label whatever the classes are.
        for container, y in _containers([np.nan, np.nan]):
            with self.subTest(container=container):
                aggregated = majority_vote(_annotator_column(y))
                self.assertEqual(aggregated.shape, (len(y),))
                self.assertTrue(is_unlabeled(aggregated).all())

    def test_inferring_a_float_vocabulary_from_mixed_numbers(self):
        # The list `[0, 1.5]` becomes a floating-point array, so `(0.0, 1.5)`
        # is inferred from it. The equivalent object array carries no dtype
        # to infer from and mixes an integer with a float, so it is rejected.
        # `SklearnClassifier` is left out: scikit-learn estimators reject
        # continuous targets themselves, whatever the contract allows.
        y = [0, 1.5] * REPEATS
        for targets in (y, np.asarray(y)):
            target_spec = resolve_target_spec(
                targets, task="classification", missing_label=np.nan
            )
            self.assertEqual(target_spec.classes, (0.0, 1.5))
            ParzenWindowClassifier(missing_label=np.nan).fit(
                _samples(targets), targets
            )
        with self.assertRaisesRegex(TypeError, "one label family"):
            resolve_target_spec(
                np.asarray(y, dtype=object),
                task="classification",
                missing_label=np.nan,
            )


def _multilabel_components(classes, missing_label):
    """Return every component consuming multi-label classification labels."""

    def parzen_window_classifier(X, y):
        ParzenWindowClassifier(
            classes=classes,
            missing_label=missing_label,
            target_type="multi-label",
        ).fit(X, y)

    def sklearn_classifier(X, y):
        SklearnClassifier(
            MultiOutputClassifier(LogisticRegression()),
            classes=classes,
            missing_label=missing_label,
            target_type="multi-label",
        ).fit(X, y)

    return {
        "is_unlabeled": (
            LABELS,
            lambda X, y: is_unlabeled(
                y, missing_label, target_type="multi-label"
            ),
        ),
        "resolve_target_spec": (
            CLASSES,
            lambda X, y: resolve_target_spec(
                y,
                task="classification",
                target_type="multi-label",
                classes=classes,
                missing_label=missing_label,
            ),
        ),
        "ExtLabelEncoder": (
            CLASSES,
            lambda X, y: ExtLabelEncoder(
                classes=classes,
                missing_label=missing_label,
                target_type="multi-label",
            ).fit_transform(y),
        ),
        "ParzenWindowClassifier": (CLASSES, parzen_window_classifier),
        "SklearnClassifier": (CLASSES, sklearn_classifier),
    }


class TestMultiLabelContract(_ContractCase):
    """The rows of the multi-label classification outcome table."""

    ROWS = (
        Row([[0, 1], [2, 3]], np.nan, [[0, 2], [np.nan, np.nan]], None, ""),
        Row(
            [[0, 1], [2, 3]],
            np.nan,
            [[0, 2], [0, np.nan]],
            ValueError,
            "only `missing_label` values",
        ),
        Row(
            [["no", "yes"], ["off", "on"]],
            None,
            [["no", "off"], [None, None]],
            None,
            "",
        ),
        Row(
            [[0, 1], [0.0, 1.0]],
            np.nan,
            [[0, 1], [np.nan, np.nan]],
            ValueError,
            "one dtype across all label outputs",
            scope=CLASSES,
        ),
        Row(
            [[0, 1, 2], [0, 1]],
            np.nan,
            [[0, 1], [np.nan, np.nan]],
            ValueError,
            "exactly two",
            scope=CLASSES,
        ),
    )

    def test_rows(self):
        for row in self.ROWS:
            components = _multilabel_components(row.classes, row.missing_label)
            for name, (evidence, run) in components.items():
                if name in row.skip:
                    continue
                expected = (
                    row
                    if row.scope == LABELS or evidence == CLASSES
                    else row._replace(error=None, message="")
                )
                self.assert_row(
                    name, expected, run, _multilabel_containers(row.y)
                )

    def test_inferring_binary_vocabularies_needs_both_classes(self):
        # `classes=None` derives one binary vocabulary per column, so a
        # column observing one category supplies no vocabulary.
        y = [[0, 0], [1, 0]] * REPEATS
        for targets in (y, np.asarray(y), np.asarray(y, dtype=object)):
            with self.subTest(container=type(targets).__name__):
                with self.assertRaisesRegex(ValueError, "exactly two"):
                    resolve_target_spec(
                        targets,
                        task="classification",
                        target_type="multi-label",
                        missing_label=np.nan,
                    )


def _regression_components(missing_label):
    """Return every component consuming numerical labels."""
    return {
        "is_unlabeled": lambda X, y: is_unlabeled(y, missing_label),
        "resolve_target_spec": lambda X, y: resolve_target_spec(
            y, task="regression", missing_label=missing_label
        ),
        "SklearnRegressor": lambda X, y: SklearnRegressor(
            LinearRegression(), missing_label=missing_label
        ).fit(X, y),
        "NICKernelRegressor": lambda X, y: NICKernelRegressor(
            missing_label=missing_label
        ).fit(X, y),
        "RandomSampling": lambda X, y: RandomSampling(
            missing_label=missing_label
        ).query(X, y, candidates=X),
    }


class TestRegressionContract(_ContractCase):
    """The rows of the regression outcome table, which declares no classes."""

    ROWS = (
        RegressionRow(np.nan, [0, 1.5, np.nan], None, ""),
        RegressionRow(None, [0, 1.5, None], None, ""),
        RegressionRow(-999, [0, 1.5, -999], None, ""),
        # The message names what the container holds: a NumPy array of
        # `[0, 1.5, "?"]` holds three strings and no numerical label at
        # all,
        # whereas a list or an object array holds a string missing label
        # beside two numerical labels. Both are rejected. `RandomSampling`
        # serves both tasks and is not told which one applies, so it reads
        # that array as the string categories it also is; see the test
        # below.
        RegressionRow(
            "?",
            [0, 1.5, "?"],
            TypeError,
            "",
            skip=("RandomSampling", "is_unlabeled"),
        ),
        RegressionRow(None, [0, np.nan, None], ValueError, "contains NaN"),
    )

    def test_rows(self):
        for row in self.ROWS:
            expected = Row(
                None, row.missing_label, row.y, row.error, row.message
            )
            components = _regression_components(row.missing_label)
            for name, run in components.items():
                if name in row.skip:
                    continue
                self.assert_row(name, expected, run, _containers(row.y))

    def test_a_task_agnostic_strategy_reads_strings_as_categories(self):
        # A NumPy array of `[0, 1.5, "?"]` holds strings, so a strategy that
        # is not told which task applies accepts it as string categories
        # with a string missing label. A list or an object array keeps the
        # numbers, which no string missing label can denote a missing one
        # of.
        y = [0, 1.5, "?"] * REPEATS
        X = _samples(y)
        RandomSampling(missing_label="?").query(X, np.asarray(y), candidates=X)
        for targets in (y, np.asarray(y, dtype=object)):
            with self.subTest(container=type(targets).__name__):
                with self.assertRaisesRegex(
                    TypeError, "is not compatible with"
                ):
                    RandomSampling(missing_label="?").query(
                        X, targets, candidates=X
                    )

    def test_no_class_vocabulary_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "not accepted for regression"):
            resolve_target_spec(
                [0.0, 1.5], task="regression", classes=[0.0, 1.5]
            )

    def test_boolean_labels_are_rejected(self):
        # The contract knows numerical labels only, and a Boolean
        # array is not a numerical one however its values compare.
        y = np.array([True, False] * REPEATS)
        X = _samples(y)
        for targets in (y, list(y), y.astype(object)):
            with self.subTest(container=type(targets).__name__):
                with self.assertRaisesRegex(TypeError, "numerical labels"):
                    SklearnRegressor(
                        LinearRegression(), missing_label=np.nan
                    ).fit(X, targets)


class TestDocumentedOutcomeTables(unittest.TestCase):
    """The documented outcome rows and the swept cases, counted."""

    DOCUMENT = Path(__file__).parents[3] / "docs" / "target_semantics.rst"

    TABLES = {
        "Single-output classification outcomes": TestSingleOutputContract,
        "Multi-label classification outcomes": TestMultiLabelContract,
        "Regression outcomes, with no class vocabulary": (
            TestRegressionContract
        ),
    }

    # The documented row `classes=None`, `missing_label=NaN`, `y=[0, 1.5]`
    # is the one row whose verdict depends on the container: the list
    # becomes a floating-point array and is accepted, whereas the object
    # array of the same values mixes families and is rejected. Every case of
    # the sweep runs in all three containers, so
    # `test_inferring_a_float_vocabulary_from_mixed_numbers` covers that row
    # by naming its containers instead.
    ROWS_WITHOUT_A_SWEPT_CASE = {"Single-output classification outcomes": 1}

    def _documented_rows(self, title):
        """Count the outcome rows the document lists for one table."""
        lines = self.DOCUMENT.read_text().splitlines()
        marker = f".. list-table:: {title}"
        self.assertIn(marker, lines)
        rows = 0
        for line in lines[lines.index(marker) + 1 :]:
            if line.strip() and not line.startswith(" "):
                break
            rows += bool(re.match(r"\s+\* - ", line))
        # Every outcome table declares `:header-rows: 1`.
        return rows - 1

    def test_every_documented_outcome_row_is_swept(self):
        for title, case in self.TABLES.items():
            with self.subTest(table=title):
                self.assertEqual(
                    self._documented_rows(title),
                    len(case.ROWS)
                    + self.ROWS_WITHOUT_A_SWEPT_CASE.get(title, 0),
                    msg=f"The table '{title}' and `{case.__name__}.ROWS` "
                    "must describe the same outcomes. Add the missing row "
                    "or case, or record the row as an exception with the "
                    "test covering it.",
                )


class TestMissingLabelContract(_ContractCase):
    """The rows of the missing-label compatibility table.

    A compatible missing label is checked with it present among the labels,
    against every component the labels reach: a numeric missing label widens
    the labels it is stored beside, and a component resolving a class
    vocabulary from them has to recognize the widened values as the declared
    classes. An incompatible missing label is checked with it absent,
    because it could not denote anything there in the first place.
    """

    NUMERIC_CLASSES = [0, 1]
    BOOLEAN_CLASSES = [False, True]
    STRING_CLASSES = ["cat", "dog"]
    MEASUREMENTS = [0.5, 1.5]

    def _assert_categories(self, classes, missing_label, accepted):
        y = (
            [classes[0], missing_label, classes[1]]
            if accepted
            else list(classes)
        )
        row = Row(
            classes,
            missing_label,
            y,
            None if accepted else TypeError,
            "" if accepted else "is not compatible with",
        )
        components = _single_output_components(classes, missing_label)
        for name, (_, run) in components.items():
            self.assert_row(name, row, run, _containers(y))

    def _assert_numerical_labels(self, missing_label, accepted):
        y = (
            [self.MEASUREMENTS[0], missing_label, self.MEASUREMENTS[1]]
            if accepted
            else list(self.MEASUREMENTS)
        )
        row = Row(
            None,
            missing_label,
            y,
            None if accepted else TypeError,
            "" if accepted else "is not compatible with",
        )
        for name, run in _regression_components(missing_label).items():
            self.assert_row(name, row, run, _containers(y))

    def test_numeric_and_boolean_categories(self):
        for classes in (self.NUMERIC_CLASSES, self.BOOLEAN_CLASSES):
            with self.subTest(classes=classes):
                for missing_label in (None, np.nan, -2.5):
                    self._assert_categories(classes, missing_label, True)
                self._assert_categories(classes, "?", False)

    def test_string_categories(self):
        for missing_label in (None, "?"):
            self._assert_categories(self.STRING_CLASSES, missing_label, True)
        for missing_label in (np.nan, -2.5):
            self._assert_categories(self.STRING_CLASSES, missing_label, False)

    def test_numerical_labels(self):
        for missing_label in (None, np.nan, -2.5):
            self._assert_numerical_labels(missing_label, True)
        self._assert_numerical_labels("?", False)

    def test_infinite_values_are_no_missing_labels(self):
        for missing_label in (np.inf, -np.inf):
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    is_unlabeled(self.NUMERIC_CLASSES, missing_label)

    def test_a_missing_label_is_never_a_class(self):
        for classes, missing_label in (
            ([0, 1], 1),
            ([0.0, 1.5], 1.5),
            (["cat", "dog"], "cat"),
        ):
            with self.subTest(classes=classes):
                with self.assertRaisesRegex(
                    ValueError, "contains `missing_label="
                ):
                    resolve_target_spec(
                        classes,
                        task="classification",
                        classes=classes,
                        missing_label=missing_label,
                    )


class TestClassIdentityRoundTrip(_ContractCase):
    """Exact class identity across the encode-fit-predict round trip.

    The contract promises that a declared class label is returned as the
    same value it was declared as, however wide it is. The guarantee is
    owned by shared code, not by any one estimator: `_as_label_array` keeps
    exact integers out of a lossy float array, `_lossless_decode_dtype`
    chooses the storage dtype, `ExtLabelEncoder` maps classes to codes, and
    `SkactivemlClassifier._decode_class_labels` narrows predictions back to
    the declared class dtype.

    `ParzenWindowClassifier` is the vehicle rather than the subject. Its
    `predict_proba` on well-separated samples is analytically `np.eye(2)`,
    which is what allows an exact assertion on the predicted values instead
    of the dtype-only check the estimator templates can make. It stays the
    only estimator exercised here, wrapped where a wrapper is on the path.
    """

    LARGE_LABELS = [2**53, 2**53 + 1]

    def test_raw_list_fit_preserves_integer_class_identity(self):
        for labels in (
            [2**53, 2**53 + 1],
            [2**63, 2**63 + 1],
        ):
            for wrapped in (False, True):
                for declared in (False, True):
                    with self.subTest(
                        labels=repr(labels), wrapped=wrapped, declared=declared
                    ):
                        classes = labels if declared else None
                        clf = ParzenWindowClassifier(classes=classes)
                        if wrapped:
                            clf = SklearnClassifier(clf, classes=classes)
                        clf.fit([[0.0], [100.0], [50.0]], [*labels, np.nan])
                        self.assertEqual(
                            clf.predict([[0.0], [100.0]]).tolist(), labels
                        )

    def test_raw_list_multilabel_fit_preserves_integer_class_identity(self):
        labels = self.LARGE_LABELS
        clf = ParzenWindowClassifier(classes=[labels, labels])
        clf.fit(
            [[0.0], [100.0], [50.0]],
            [[labels[0], labels[0]], [labels[1], labels[1]], [np.nan, np.nan]],
        )
        self.assertEqual(
            clf.predict([[0.0], [100.0]]).tolist(),
            [[labels[0], labels[0]], [labels[1], labels[1]]],
        )

    def test_predictions_preserve_large_integer_class_identity(self):
        for classes in (
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            np.array([-(2**63), 2**63 - 1], dtype=np.int64),
            np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
        ):
            for wrapped in (False, True):
                for cost_matrix in (None, [[1, 0], [0, 1]]):
                    with self.subTest(
                        classes=repr(classes),
                        wrapped=wrapped,
                        cost_matrix=cost_matrix,
                    ):
                        X = np.array([[0.0], [100.0]])
                        estimator = ParzenWindowClassifier(classes=classes)
                        clf = (
                            SklearnClassifier(estimator, classes=classes)
                            if wrapped
                            else estimator
                        )
                        clf.set_params(cost_matrix=cost_matrix)
                        clf.fit(X, classes)
                        np.testing.assert_array_equal(
                            clf.predict_proba(X), np.eye(2)
                        )
                        predicted = clf.predict(X)
                        self.assertEqual(predicted.dtype, classes.dtype)
                        self.assertEqual(
                            predicted.tolist(),
                            (
                                classes.tolist()
                                if cost_matrix is None
                                else classes[::-1].tolist()
                            ),
                        )

    def test_multilabel_predictions_preserve_large_integer_class_identity(
        self,
    ):
        for classes in (
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
        ):
            for wrapped in (False, True):
                with self.subTest(classes=repr(classes), wrapped=wrapped):
                    y = np.column_stack([classes, classes[::-1]])
                    clf = ParzenWindowClassifier(classes=[classes, classes])
                    if wrapped:
                        clf = SklearnClassifier(
                            MultiOutputClassifier(
                                ParzenWindowClassifier(classes=classes)
                            ),
                            classes=[classes, classes],
                        )
                    clf.fit([[0.0], [100.0]], y)
                    predicted = clf.predict([[0.0], [100.0]])
                    self.assertEqual(predicted.dtype, y.dtype)
                    self.assertEqual(predicted.tolist(), y.tolist())
