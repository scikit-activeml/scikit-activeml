"""Representations and checks of class labels and missing labels."""

import numpy as np

_NO_MISSING_LABEL = object()

_SUPPORTED_LABELS = "Labels must be strings or real numbers."


def _as_label_array(y):
    """Convert labels to an array without turning numbers into strings.

    Parameters
    ----------
    y : array-like
        Original labels. Existing arrays retain their dtype and values.

    Returns
    -------
    y_array : numpy.ndarray
        The ordinary NumPy representation, or an object array of the original
        values if NumPy would turn values other than strings into strings,
        e.g., for `["cat", np.nan]`. Shape validation is left to the caller.
    """
    result = np.asarray(y)
    if isinstance(y, (list, tuple)) and result.dtype.kind == "U":
        original = np.asarray(y, dtype=object)
        if not all(isinstance(value, str) for value in original.flat):
            return original
    return result


def _as_class_vocabulary_array(classes):
    """Convert one class vocabulary to an array.

    Parameters
    ----------
    classes : iterable of scalar labels
        One class vocabulary.

    Returns
    -------
    classes_array : numpy.ndarray of shape (n_classes,)
        The vocabulary as converted by `_as_label_array`.
    """
    return _as_label_array(list(classes))


def _common_label_dtype(arrays, missing_label=_NO_MISSING_LABEL):
    """Choose one dtype for labels and, optionally, the missing label.

    Parameters
    ----------
    arrays : sequence of array-like
        Arrays of labels.
    missing_label : scalar or str or None, optional
        Also represent this missing label if supplied. Omission selects a
        label-only dtype. Explicitly passing `None` includes that value.

    Returns
    -------
    dtype : numpy.dtype
        NumPy's common dtype, or `object` for `None` and for strings mixed
        with numbers.
    """
    dtypes = [np.asarray(array).dtype for array in arrays]
    if missing_label is not _NO_MISSING_LABEL:
        dtypes.append(np.asarray([missing_label]).dtype)
    kinds = {dtype.kind for dtype in dtypes}
    if "O" in kinds or ("U" in kinds and len(kinds) > 1):
        return np.dtype(object)
    return np.result_type(*dtypes)


def _missing_mask_and_kind(y, missing_label, *, name, numerical=False):
    """Locate missing labels and check the observed ones.

    Parameters
    ----------
    y : numpy.ndarray
        Labels including missing values.
    missing_label : scalar or str or None
        Value to represent a missing label, or `_NO_MISSING_LABEL` if `y`
        cannot contain one, e.g., a class vocabulary.
    name : str
        The name of the checked variable, used in error messages.
    numerical : bool, default=False
        Whether the labels must be real numbers other than Boolean values.

    Returns
    -------
    is_missing : numpy.ndarray of shape `y.shape`
        Boolean mask locating `missing_label` in `y`.
    kind : "str" or "number" or None
        The kind of the observed labels, or `None` if `y` holds none.

    Raises
    ------
    TypeError
        If an observed label is neither a string nor a real number, if the
        observed labels mix strings and numbers, or if they are incompatible
        with `missing_label`.
    ValueError
        If an observed number is nonfinite.
    """
    if missing_label is not _NO_MISSING_LABEL:
        _check_missing_label_value(missing_label)
    if y.dtype.kind == "O":
        # Entries of other types may not compare with the missing label as
        # scalars, so they are rejected before the comparison.
        for value_type in dict.fromkeys(map(type, y.flat)):
            if value_type is not type(None):
                _kind_of_type(value_type, name=name)
        is_missing = _object_missing_mask(y, missing_label)
        observed = y[~is_missing]
        types = dict.fromkeys(map(type, observed))
    else:
        is_missing = _typed_missing_mask(y, missing_label)
        observed = y[~is_missing]
        types = [y.dtype.type] if observed.size else []
    kind = _kind_of_types(types, name=name, numerical=numerical)
    if missing_label is not _NO_MISSING_LABEL:
        _check_missing_label_for_kind(missing_label, kind, name=name)
    if kind == "number" and observed.dtype.kind in "fO":
        _check_finite_labels(observed, name=name)
    return is_missing, kind


def _kind_of_types(types, *, name, numerical=False):
    """Return whether the types of observed labels are strings or numbers.

    Parameters
    ----------
    types : iterable of type
        The scalar types of the observed labels.
    name : str
        The name of the checked variable, used in error messages.
    numerical : bool, default=False
        Whether the labels must be real numbers other than Boolean values.

    Returns
    -------
    kind : "str" or "number" or None
        The kind of the labels, or `None` if `types` is empty.
    """
    kinds = set()
    for value_type in types:
        kinds.add(_kind_of_type(value_type, name=name))
        if numerical and issubclass(value_type, (str, bool, np.bool_)):
            raise TypeError(
                f"`{name}` must contain numerical labels, but contains "
                f"values of type '{value_type.__name__}'."
            )
    if len(kinds) > 1:
        raise TypeError(
            f"`{name}` mixes strings and numbers. Labels must be either all "
            "strings or all real numbers."
        )
    return kinds.pop() if kinds else None


def _kind_of_type(value_type, *, name):
    """Return whether a scalar type of labels holds strings or numbers.

    Parameters
    ----------
    value_type : type
        The scalar type of a label.
    name : str
        The name of the checked variable, used in error messages.

    Returns
    -------
    kind : "str" or "number"
        The kind of labels of type `value_type`.

    Raises
    ------
    TypeError
        If `value_type` is neither a string nor a real number type.
    """
    if issubclass(value_type, str):
        return "str"
    if _is_number_type(value_type):
        return "number"
    if value_type is type(None):
        raise TypeError(
            f"`{name}` contains `None`, which is only valid as the "
            "configured `missing_label`."
        )
    raise TypeError(
        f"`{name}` contains values of the unsupported type "
        f"'{value_type.__name__}'. {_SUPPORTED_LABELS}"
    )


def _is_number_type(value_type):
    """Return whether a scalar type holds real numbers."""
    # NumPy's timedelta64 subclasses its integer type.
    return issubclass(
        value_type, (int, float, np.bool_, np.integer, np.floating)
    ) and not issubclass(value_type, np.timedelta64)


def _label_kind(values, *, name):
    """Return whether class labels or observed labels are strings or numbers.

    Parameters
    ----------
    values : array-like
        Class labels or observed labels, without missing values.
    name : str
        The name of the checked variable, used in error messages.

    Returns
    -------
    kind : "str" or "number" or None
        The kind of the labels, or `None` if `values` is empty.
    """
    _, kind = _missing_mask_and_kind(
        np.asarray(values), _NO_MISSING_LABEL, name=name
    )
    return kind


def _check_compatible_kinds(observed, declared, *, name):
    """Check that observed labels and declared classes share one kind.

    Parameters
    ----------
    observed : array-like
        The observed labels, without missing values.
    declared : array-like
        The class vocabulary of one output.
    name : str
        The name of the vocabulary, used in error messages.

    Raises
    ------
    TypeError
        If one of them holds strings and the other numbers.
    """
    observed_kind = _label_kind(observed, name="y")
    declared_kind = _label_kind(
        np.asarray(list(declared), dtype=object), name=name
    )
    if observed_kind is not None and observed_kind != declared_kind:
        raise TypeError(
            f"The labels in `y` are not type-compatible with {name}."
        )


def _check_missing_label_value(missing_label):
    """Check that a missing label is a supported scalar value.

    Parameters
    ----------
    missing_label : scalar or str or None
        Value to represent a missing label.

    Raises
    ------
    TypeError
        If `missing_label` is neither a real number other than a Boolean
        value, nor a string, nor `None`.
    ValueError
        If `missing_label` is infinite.
    """
    if missing_label is None or isinstance(missing_label, str):
        return
    if isinstance(missing_label, (bool, np.bool_)) or not _is_number_type(
        type(missing_label)
    ):
        raise TypeError(
            f"`missing_label` has type '{type(missing_label).__name__}', but "
            "must be a real number, a string, `np.nan`, or `None`."
        )
    if isinstance(missing_label, (float, np.floating)) and np.isinf(
        missing_label
    ):
        raise ValueError(
            f"`missing_label={missing_label}` must be finite. NaN marks "
            "a missing label; an infinity is neither a label nor a "
            "missing one."
        )


def _check_missing_label_for_kind(missing_label, kind, *, name=None):
    """Check that a missing label is compatible with a kind of labels.

    Parameters
    ----------
    missing_label : scalar or str or None
        Value to represent a missing label.
    kind : "str" or "number" or None
        The kind of the labels, or `None` if no label provides evidence
        about it.
    name : str, default=None
        The name of the variable holding the labels, used in error messages.

    Raises
    ------
    TypeError
        If the missing label is a number and the labels are strings, or the
        other way around.
    """
    _check_missing_label_value(missing_label)
    if missing_label is None or kind is None:
        return
    name = "target object" if name is None else str(name)
    if kind == "str" and not isinstance(missing_label, str):
        raise TypeError(
            f"`missing_label={missing_label!r}` is a number and is not "
            f"compatible with the string labels in `{name}`. Use a string "
            "missing label or `None` instead, and mark the missing entries "
            "with it."
        )
    if kind == "number" and isinstance(missing_label, str):
        raise TypeError(
            f"`missing_label={missing_label!r}` is a string and is not "
            f"compatible with the numeric labels in `{name}`. Use `np.nan`, "
            "a numeric missing label, or `None` instead."
        )


def _dtype_kind(target_type):
    """Map the dtype or scalar type of labels to their kind.

    Parameters
    ----------
    target_type : numpy.dtype or type or tuple
        The dtype or scalar type of the labels a missing label has to fit.

    Returns
    -------
    kind : "str" or "number" or None
        The kind of the labels, or `None` if `target_type` constrains no
        missing label, e.g., `object`.
    """
    try:
        dtype = np.dtype(target_type)
    except TypeError:
        return None
    if dtype.kind in "SU":
        return "str"
    return "number" if dtype.kind in "biuf" else None


def _typed_missing_mask(y, missing_label):
    """Locate a missing label in an array of a non-object dtype.

    Parameters
    ----------
    y : numpy.ndarray
        Labels of a non-object dtype, including missing values.
    missing_label : scalar or str or None
        Value to represent a missing label, or `_NO_MISSING_LABEL`.

    Returns
    -------
    is_missing : numpy.ndarray of shape `y.shape`
        Boolean mask locating `missing_label` in `y`.
    """
    if isinstance(missing_label, str):
        if y.dtype.kind == "U":
            return y == missing_label
    elif _is_nan_missing_label(missing_label):
        if y.dtype.kind == "f":
            return np.isnan(y)
    elif (
        missing_label is not None
        and missing_label is not _NO_MISSING_LABEL
        and y.dtype.kind in "biuf"
    ):
        # A Python scalar would be cast to the dtype of `y`, so that, e.g.,
        # the missing label 16777217 would match the `float32` label
        # 16777216.
        return y == np.asarray(missing_label)
    return np.zeros(y.shape, dtype=bool)


def _object_missing_mask(values, missing_label):
    """Locate a missing label among the entries of an object array.

    Parameters
    ----------
    values : numpy.ndarray of dtype object
        Labels including missing values.
    missing_label : scalar or str or None
        Value to represent a missing label, or `_NO_MISSING_LABEL`.

    Returns
    -------
    is_missing : numpy.ndarray of shape `values.shape`
        Boolean mask locating `missing_label` in `values`. Numeric NaN
        matches neither the string `"nan"` nor `None`.
    """
    if _is_nan_missing_label(missing_label):
        # NaN is the only supported label that differs from itself.
        return values != values
    return values == missing_label


def _matches_missing_label(value, missing_label):
    """Check whether one scalar value is the configured missing label."""
    values = np.empty(1, dtype=object)
    values[0] = value
    return bool(_object_missing_mask(values, missing_label)[0])


def _holds_missing_label(values, missing_label):
    """Check whether the dtype of labels also holds the missing label.

    Parameters
    ----------
    values : numpy.ndarray
        The label array the missing label would be stored in.
    missing_label : scalar or str or None
        Value to represent a missing label.

    Returns
    -------
    holds_missing_label : bool
        Whether `missing_label`, stored in `values.dtype`, is still
        recognized as the missing label, e.g., `np.nan` in `float32` or `-1`
        in `int8`, but not `0.1` in `float32`.
    """
    try:
        stored = np.array([missing_label], dtype=values.dtype)
    except (TypeError, ValueError, OverflowError):
        return False
    if stored.dtype.kind == "O":
        return True
    return bool(_typed_missing_mask(stored, missing_label)[0])


def _missing_label_kind(missing_label):
    """Return the kind of labels a missing label can mark as missing.

    Parameters
    ----------
    missing_label : scalar or str or None
        Value to represent a missing label.

    Returns
    -------
    kind : "number" or "str" or None
        The kind of labels `missing_label` can mark a missing one of, or
        `None` if it can mark one of every kind.
    """
    if missing_label is None:
        return None
    return "str" if isinstance(missing_label, str) else "number"


def _is_nan_missing_label(missing_label):
    """Return whether a missing label is a floating-point NaN."""
    return isinstance(missing_label, (float, np.floating)) and bool(
        np.isnan(missing_label)
    )


def _check_finite_labels(values, *, name):
    """Check that numeric labels are finite.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric labels without missing values.
    name : str
        The name of the checked variable, used in error messages.

    Raises
    ------
    ValueError
        If any value is NaN or infinite.
    """
    values = np.asarray(values, dtype=float)
    is_finite = np.isfinite(values)
    if is_finite.all():
        return
    if np.isnan(values[~is_finite]).any():
        raise ValueError(
            f"`{name}` contains NaN, which is only valid as the configured "
            "`missing_label`."
        )
    raise ValueError(
        f"`{name}` contains an infinite value, which is neither a label nor "
        "a missing label."
    )
