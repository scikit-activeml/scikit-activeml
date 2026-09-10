import inspect
import numbers
from collections.abc import Mapping, Sequence

import numpy as np

from ..classifier import ParzenWindowClassifier
from ..pool import uncertainty_scores

# The public attributes a pool query strategy commits while validating.
QUERY_STATE_ATTRIBUTES = (
    "n_features_in_",
    "missing_label_",
    "random_state_",
)

# Stream query strategies expose no `missing_label_`, but commit the budget
# state their budget manager is built from.
STREAM_QUERY_STATE_ATTRIBUTES = (
    "n_features_in_",
    "random_state_",
    "budget_",
    "budget_manager_",
)


def assert_no_query_state(
    test_case, strategy, attributes=QUERY_STATE_ATTRIBUTES
):
    """Assert that semantic query failure did not commit public state.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case providing the assertion.
    strategy : skactiveml query strategy
        The query strategy whose `query` failed.
    attributes : iterable of str, default=QUERY_STATE_ATTRIBUTES
        The attributes a failed query must not have committed. Stream query
        strategies pass `STREAM_QUERY_STATE_ATTRIBUTES` instead, because they
        commit other state.
    """
    for attribute in attributes:
        test_case.assertFalse(
            hasattr(strategy, attribute),
            msg=(
                f"{type(strategy).__name__} committed query state "
                f"`{attribute}` after semantic failure."
            ),
        )


def state_difference(actual, expected, name="state"):
    """Describe the first difference between two object states.

    The comparison is recursive and by value: objects are compared through
    their attributes, mappings and sequences element by element, and numeric
    arrays and scalars with equal-NaN semantics, so that two states holding
    `nan` in the same place count as equal. Random generators are compared
    through their drawn-from state, so that an advanced generator counts as a
    difference.

    Objects are only comparable when they have the same type. An object the
    recursion cannot decompose falls back on `==` where its type defines one,
    and counts as equal otherwise, because two copies of a state without
    attributes and without value equality, e.g. a function, are
    indistinguishable here.

    Parameters
    ----------
    actual : object
        The state to check, e.g. the estimator a query was given.
    expected : object
        The state to check against, e.g. a copy taken before the query.
    name : str, default="state"
        Name of the compared state, used to root the reported path.

    Returns
    -------
    difference : str or None
        Human-readable description of the first difference found, naming the
        path to it, or `None` if both states are equal.
    """
    return _state_difference(actual, expected, name, set())


def assert_state_unchanged(test_case, actual, expected, name="state", msg=""):
    """Assert that two object states are equal by value.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case reporting the failure.
    actual : object
        The state to check, e.g. the estimator a query was given.
    expected : object
        The state to check against, e.g. a copy taken before the query.
    name : str, default="state"
        Name of the compared state, used to root the reported path.
    msg : str, default=""
        Message prefixed to the reported difference.
    """
    difference = state_difference(actual, expected, name=name)
    test_case.assertIsNone(
        difference, msg=f"{msg} {difference}" if msg else difference
    )


def _state_difference(actual, expected, path, seen):
    """Return the first difference between two states, or `None`."""
    if actual is expected:
        return None
    if type(actual) is not type(expected):
        return (
            f"`{path}` changed type from `{type(expected).__name__}` to "
            f"`{type(actual).__name__}`."
        )
    # Guard against the cycles an estimator referring back to its owner
    # creates. A pair under comparison is equal unless a difference is found
    # elsewhere, which is what the enclosing call reports.
    pair = (id(actual), id(expected))
    if pair in seen:
        return None
    seen.add(pair)

    if isinstance(actual, (str, bytes)):
        return _compare_values(actual == expected, actual, expected, path)
    if isinstance(actual, np.random.RandomState):
        return _state_difference(
            actual.get_state(), expected.get_state(), f"{path}.state", seen
        )
    if isinstance(actual, np.random.Generator):
        return _state_difference(
            actual.bit_generator.state,
            expected.bit_generator.state,
            f"{path}.bit_generator.state",
            seen,
        )
    if isinstance(actual, np.ndarray):
        return _array_difference(actual, expected, path, seen)
    if isinstance(actual, (numbers.Number, np.bool_)):
        return _compare_values(
            _scalars_equal(actual, expected), actual, expected, path
        )
    if isinstance(actual, Mapping):
        return _mapping_difference(actual, expected, path, seen)
    if isinstance(actual, (set, frozenset)):
        return _compare_values(actual == expected, actual, expected, path)
    if isinstance(actual, Sequence):
        return _sequence_difference(actual, expected, path, seen)

    attributes = _attribute_state(actual)
    expected_attributes = _attribute_state(expected)
    if attributes is not None or expected_attributes is not None:
        return _mapping_difference(
            attributes or {}, expected_attributes or {}, path, seen
        )
    if type(actual).__eq__ is not object.__eq__:
        return _compare_values(
            _equal_or_identical(actual, expected), actual, expected, path
        )
    # Two objects of the same type without discoverable state and without
    # value equality, e.g. a function, are indistinguishable here. Comparing
    # them by identity would report every copy as a difference.
    return None


def _array_difference(actual, expected, path, seen):
    """Return the first difference between two arrays, or `None`."""
    if actual.shape != expected.shape:
        return (
            f"`{path}` changed shape from {expected.shape} to {actual.shape}."
        )
    if actual.dtype != expected.dtype:
        return (
            f"`{path}` changed dtype from `{expected.dtype}` to "
            f"`{actual.dtype}`."
        )
    if actual.dtype.kind == "O":
        # Object arrays hold arbitrary values, which only the general
        # recursion can compare.
        for index, (value, expected_value) in enumerate(
            zip(actual.ravel(), expected.ravel())
        ):
            difference = _state_difference(
                value, expected_value, f"{path}.ravel()[{index}]", seen
            )
            if difference is not None:
                return difference
        return None
    equal = _elementwise_equal(actual, expected)
    if equal is None:
        return _compare_values(
            _equal_or_identical(actual, expected), actual, expected, path
        )
    if equal.all():
        return None
    index = np.unravel_index(np.argmax(~equal), equal.shape)
    position = "".join(f"[{axis}]" for axis in index)
    return (
        f"`{path}{position}` changed from {expected[index]!r} to "
        f"{actual[index]!r}."
    )


def _elementwise_equal(actual, expected):
    """Compare two arrays element by element, treating `nan` as equal."""
    try:
        with np.errstate(invalid="ignore"):
            equal = np.asarray(actual == expected)
        if actual.dtype.kind in "fc":
            equal = equal | (np.isnan(actual) & np.isnan(expected))
    except (TypeError, ValueError):
        return None
    if equal.dtype != bool or equal.shape != actual.shape:
        return None
    return equal


def _mapping_difference(actual, expected, path, seen):
    """Return the first difference between two mappings, or `None`."""
    missing = [key for key in expected if key not in actual]
    if missing:
        return f"`{_key_path(path, missing[0])}` is missing."
    added = [key for key in actual if key not in expected]
    if added:
        return f"`{_key_path(path, added[0])}` was added."
    for key in expected:
        difference = _state_difference(
            actual[key], expected[key], _key_path(path, key), seen
        )
        if difference is not None:
            return difference
    return None


def _sequence_difference(actual, expected, path, seen):
    """Return the first difference between two sequences, or `None`."""
    if len(actual) != len(expected):
        return (
            f"`{path}` changed length from {len(expected)} to {len(actual)}."
        )
    for index, (value, expected_value) in enumerate(zip(actual, expected)):
        difference = _state_difference(
            value, expected_value, f"{path}[{index}]", seen
        )
        if difference is not None:
            return difference
    return None


def _key_path(path, key):
    """Extend a state path by a mapping key or an attribute name."""
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _attribute_state(obj):
    """Return the attributes holding an object's state, or `None`.

    An object storing its state in `__slots__` or in a C extension exposes it
    through `__getstate__` rather than through `__dict__`. An object with no
    attributes at all reports `None`, so that the caller can fall back on
    value equality instead of accepting an empty state as equal.
    """
    state = getattr(obj, "__dict__", None)
    if state:
        return dict(state)
    try:
        state = obj.__getstate__()
    except (AttributeError, TypeError):
        return None
    return dict(state) if isinstance(state, Mapping) and state else None


def _scalars_equal(actual, expected):
    """Compare two numbers, treating `nan` as equal to `nan`."""
    try:
        actual_nan, expected_nan = bool(np.isnan(actual)), bool(
            np.isnan(expected)
        )
    except (TypeError, ValueError):
        return _equal_or_identical(actual, expected)
    if actual_nan or expected_nan:
        return actual_nan and expected_nan
    return _equal_or_identical(actual, expected)


def _equal_or_identical(actual, expected):
    """Compare two values by `==`, falling back to identity."""
    try:
        return bool(actual == expected)
    except Exception:
        return actual is expected


def _compare_values(equal, actual, expected, path):
    """Report a value difference unless the two values are equal."""
    if equal:
        return None
    return (
        f"`{path}` changed from {_short_repr(expected)} to "
        f"{_short_repr(actual)}."
    )


def _short_repr(value, limit=120):
    """Return a repr short enough for an assertion message."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def assert_predicts_class_dtype(test_case, y_pred, classes):
    """Assert that predictions carry the declared class dtype.

    The label encoder decodes into a dtype that can also represent
    `missing_label`, so predictions must be narrowed back to the dtype of
    the declared classes to stay usable where those labels are expected.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case providing the assertion.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The predicted class labels.
    classes : numpy.ndarray or list of numpy.ndarray
        The declared classes, i.e., one array per label output for a
        multi-label target and one array otherwise.
    """
    if isinstance(classes, (list, tuple)):
        expected_dtype = np.result_type(
            *[np.asarray(classes_j).dtype for classes_j in classes]
        )
    else:
        expected_dtype = np.asarray(classes).dtype
    test_case.assertEqual(
        np.asarray(y_pred).dtype,
        expected_dtype,
        msg="`predict` must return the declared class dtype.",
    )


def assert_attributes_unchanged(
    test_case, estimator, attributes_before, ignored=()
):
    """Assert that `estimator` holds exactly the snapshotted attributes.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case reporting the failure.
    estimator : object
        The estimator to compare against the snapshot.
    attributes_before : dict
        Snapshot of `estimator.__dict__` taken before the failing call.
    ignored : iterable of str, default=()
        Names of attributes the caller changed after taking the snapshot,
        e.g. an `estimator` replaced to make a re-fit fail.
    """
    test_case.assertEqual(
        set(estimator.__dict__) - set(ignored),
        set(attributes_before) - set(ignored),
    )
    for name, value in attributes_before.items():
        if name not in ignored:
            test_case.assertIs(estimator.__dict__[name], value)


def assert_fit_failure_is_transactional(
    test_case, estimator, action, expected_error, expected_message
):
    """Assert that a rejected fit raises as expected and commits no state.

    This is the counterpart of `assert_no_query_state` for a fit, and is
    strictly stronger: it compares the full `__dict__` by identity rather than
    three named absences, so it holds for an already fitted estimator as well
    as an unfitted one, and a same-valued replacement object fails it.

    Comparing by identity is also its one blind spot: an estimator holding
    fitted state that its fit updates in place, e.g. a sliding window, passes
    this assertion while carrying mutated contents. Assert those contents
    separately.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case reporting the failure.
    estimator : object
        The estimator whose fit is expected to be rejected. It is snapshotted
        here, so a caller that mutated it beforehand still gets the full
        identity comparison over every attribute. There is deliberately no
        `ignored` parameter for that reason: it could only weaken the
        comparison.
    action : callable
        Zero-argument callable performing the rejected fit.
    expected_error : type
        Exception type the rejection is expected to raise.
    expected_message : str
        Pattern the raised message is expected to match.
    """
    attributes_before = dict(estimator.__dict__)

    with test_case.assertRaisesRegex(expected_error, expected_message):
        action()

    assert_attributes_unchanged(test_case, estimator, attributes_before)


def check_positional_args(func, func_name, param_dict, kwargs_name=None):
    func_params = inspect.signature(func).parameters
    kwargs_var_keyword = []
    # Get kwargs variables
    kwargs_var_keyword = list(
        filter(lambda p: p.kind == p.VAR_KEYWORD, func_params.values())
    )

    # Test if each required key except for kwargs is included.
    if param_dict is not None:
        for key, val in func_params.items():
            if (
                key != "self"
                and val not in kwargs_var_keyword
                and val.default == inspect._empty
                and key not in param_dict
            ):
                if kwargs_name in None:
                    raise ValueError(
                        f"Missing positional argument `{key}` of `{func_name}`"
                        f" in `{func_name}_default_kwargs`."
                    )
                else:
                    raise ValueError(
                        f"Missing positional argument `{key}` of `{func_name}`"
                        f" in `{kwargs_name}`."
                    )


def check_test_param_test_availability(
    class_, func, func_name, not_test, logic_test=True
):
    # Get func parameters.
    func_params = inspect.signature(func).parameters
    kwargs_var_keyword = list(
        filter(lambda p: p.kind == p.VAR_KEYWORD, func_params.values())
    )

    # Check func parameters.
    for param, val in func_params.items():
        if param in not_test or val in kwargs_var_keyword:
            continue
        test_func_name = f"test_{func_name}_param_" + param
        with class_.subTest(msg=test_func_name):
            class_.assertTrue(
                hasattr(class_, test_func_name),
                msg=f"'{test_func_name}()' missing in {class_.__class__}",
            )
    if logic_test:
        # Check if func is being tested.
        with class_.subTest(msg=f"test_{func_name}"):
            class_.assertTrue(
                hasattr(class_, f"test_{func_name}"),
                msg=f"'test_{func_name}' missing in {class_.__class__}",
            )


class ParzenWindowClassifierEmbedding(ParzenWindowClassifier):
    def predict(self, X, return_embeddings=False):
        y_pred = super().predict(X)
        if not return_embeddings:
            return y_pred
        return y_pred, X

    def predict_proba(self, X, return_embeddings=False):
        probas = super().predict_proba(X)
        if not return_embeddings:
            return probas
        return probas, X


class ParzenWindowClassifierEmbeddingUncertainty(ParzenWindowClassifier):
    def predict(self, X, return_embeddings=False, return_uncertainties=False):
        out = self.predict_proba(
            X,
            return_embeddings=return_embeddings,
            return_uncertainties=return_uncertainties,
        )
        if isinstance(out, np.ndarray):
            return out.argmax(axis=-1)
        else:
            primary = out[0].argmax(axis=-1)
            return (primary,) + out[1:]

    def predict_proba(
        self, X, return_embeddings=False, return_uncertainties=False
    ):
        out = [super().predict_proba(X)]
        if return_embeddings:
            out.append(X)
        if return_uncertainties == "1d":
            out.append(uncertainty_scores(out[0], method="entropy"))
        elif return_uncertainties == "2d":
            out.append(
                uncertainty_scores(out[0], method="entropy").reshape(-1, 1)
            )
        elif return_uncertainties is False:
            pass
        else:
            raise ValueError(
                "`return_uncertainties` must be `1d` or `2d` or `False`."
            )
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


class ParzenWindowClassifierTuple(ParzenWindowClassifier):
    def predict(self, X):
        y_pred = super().predict_proba(X).argmax(axis=-1)
        return y_pred, X

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return probas, X


class ParzenWindowClassifierTriplet(ParzenWindowClassifier):
    def predict(self, X):
        probas = super().predict_proba(X)
        y_pred = probas.argmax(axis=-1)
        unc = uncertainty_scores(probas, method="entropy")
        return y_pred, unc, X

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        unc = uncertainty_scores(probas, method="entropy")
        return probas, unc, X


def _softmax_logits_from_probas(probas):
    return np.log(np.clip(probas, a_min=1e-12, a_max=1.0))


def _normalize_extra_outputs(extra_outputs):
    if extra_outputs is None:
        return []
    if isinstance(extra_outputs, str):
        return [extra_outputs]
    return list(extra_outputs)


class ParzenWindowClassifierLogitsEmbedding(ParzenWindowClassifier):
    fit_calls = 0

    @classmethod
    def reset_fit_calls(cls):
        cls.fit_calls = 0

    def fit(self, X, y, sample_weight=None):
        type(self).fit_calls += 1
        return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(
        self,
        X,
        return_embeddings=False,
        return_logits=False,
        extra_outputs=None,
    ):
        probas = super().predict_proba(X)
        logits = _softmax_logits_from_probas(probas)
        emb = np.asarray(X)
        extra_outputs = _normalize_extra_outputs(extra_outputs)
        if extra_outputs:
            out = [probas]
            for name in extra_outputs:
                if name == "logits":
                    out.append(logits)
                elif name in ["emb", "embedding", "embeddings"]:
                    out.append(emb)
                else:
                    raise ValueError(f"Unsupported extra output `{name}`.")
            return tuple(out)
        out = [probas]
        if return_logits:
            out.append(logits)
        if return_embeddings:
            out.append(emb)
        if len(out) == 1:
            return out[0]
        return tuple(out)


class ParzenWindowClassifierLogitsEmbeddingTuple(
    ParzenWindowClassifierLogitsEmbedding
):
    def predict_proba(self, X):
        probas = super().predict_proba(X)
        logits = _softmax_logits_from_probas(probas)
        return probas, logits, np.asarray(X)


class ParzenWindowClassifierWeirdTuple(ParzenWindowClassifier):
    def predict_proba(self, X, return_stuff=False):
        probas = super().predict_proba(X)
        if not return_stuff:
            return probas
        logits = _softmax_logits_from_probas(probas)
        return probas, np.asarray(X), logits


class ParzenWindowClassifierLogitsOnly(ParzenWindowClassifier):
    def predict_proba(self, X, return_logits=False):
        probas = super().predict_proba(X)
        if return_logits:
            return _softmax_logits_from_probas(probas)
        return probas
