"""Lossless representations of class labels and missing labels."""

from numbers import Integral

import numpy as np

_NO_MISSING_LABEL = object()

# The label families of the label and missing-value contract, in the order
# error messages name them.
_FAMILY_ORDER = ("bool", "int", "float", "str")

_FAMILY_DESCRIPTIONS = {
    "bool": "Boolean",
    "int": "integer",
    "float": "floating-point",
    "str": "string",
}

# The three roles labels play in the contract, each with its own rule for
# combining the families found among them.
#
# Classification vocabularies and observed labels hold exactly one family.
# Regression labels may combine integers and floating-point values, but
# neither Boolean values nor strings describe a numerical target. Public
# single-output mask helpers are task-agnostic: they accept values valid for
# either task, which adds only the integer/float mixture to the single-family
# classification domain.
_VOCABULARY = "vocabulary"
_LABELS = "labels"
_NUMERICAL_LABELS = "numerical labels"
_TASK_AGNOSTIC_LABELS = "task-agnostic labels"

_FAMILY_KINDS = {
    "bool": "numeric",
    "int": "numeric",
    "float": "numeric",
    "str": "str",
}

_DTYPE_KIND_FAMILIES = {
    "b": "bool",
    "i": "int",
    "u": "int",
    "f": "float",
    "U": "str",
}

_SUPPORTED_LABELS = (
    "Labels must be Boolean values, integers or finite floating-point "
    "values of at most 64 bits, or strings."
)

_STORABLE_INTEGERS = (
    "Integers must be representable as `int64` or as `uint64`."
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1


def _as_label_array(y):
    """Convert target sequences without rounding integers or stringifying them.

    Parameters
    ----------
    y : array-like
        Original targets. Existing arrays retain their dtype and values;
        precision already lost by a caller cannot be recovered.

    Returns
    -------
    y_array : numpy.ndarray
        The ordinary NumPy representation when lossless, otherwise an object
        array built from the original values. Shape validation is left to the
        caller.
    """
    result = np.asarray(y)
    if not isinstance(y, (list, tuple)) or result.dtype.kind not in "fcSU":
        return result
    if result.dtype.kind == "f":
        # Strictly below this boundary all integers are exactly representable.
        # Keep the boundary itself on the slow path: an integer just above it
        # can round down onto it. NaNs are common missing labels; infinities
        # still take the value-by-value path.
        integer_limit = 2 ** (np.finfo(result.dtype).nmant + 1)
        if np.all(np.isnan(result) | (np.abs(result) < integer_limit)):
            return result
    original = np.asarray(y, dtype=object)
    if result.dtype.kind in "fc":
        # Python int/float equality does not promote the integer to float.
        # NumPy scalar equality would mask the precision loss being checked.
        for value, converted in zip(original.flat, result.flat):
            if isinstance(value, Integral) and int(value) != converted.item():
                return original
    else:
        string_type = str if result.dtype.kind == "U" else bytes
        if any(not isinstance(value, string_type) for value in original.flat):
            return original
    return result


def _as_class_vocabulary_array(classes, *, name="classes"):
    """Convert one class vocabulary without changing its label family.

    NumPy promotes a mixture of signed and unsigned 64-bit integer scalars to
    ``float64``. That conversion can merge neighboring integers above the
    exact floating-point range before ``LabelEncoder`` sees them. Declared
    classes are therefore checked value by value and converted to a common
    integer dtype explicitly when ordinary NumPy promotion would leave the
    integer family.

    Parameters
    ----------
    classes : iterable of scalar labels
        One validated or not-yet-validated class vocabulary.
    name : str, default="classes"
        Name used in validation errors.

    Returns
    -------
    classes_array : numpy.ndarray of shape (n_classes,)
        A lossless ordinary array suitable for ``LabelEncoder``.
    """
    values = list(classes)
    exact = np.asarray(values, dtype=object)
    family = _label_family(exact, name=name, role=_VOCABULARY)
    result = np.asarray(values)

    if family == "int" and result.dtype.kind not in "iu":
        integer_values = [int(value) for value in values]
        dtype = np.uint64 if max(integer_values) > _INT64_MAX else np.int64
        result = np.asarray(integer_values, dtype=dtype)
    return result


def _lossless_decode_dtype(vocabularies, missing_label=_NO_MISSING_LABEL):
    """Choose a common dtype without changing class or missing-label values.

    Numeric conversions are checked by casting back to each original dtype
    before comparing. This avoids both Python loops over class vocabularies
    and integer-to-float promotion during equality checks. Numeric missing
    labels must never be converted into strings, even if they could be
    parsed back.

    Parameters
    ----------
    vocabularies : sequence of numpy.ndarray
        One-dimensional arrays of class labels.
    missing_label : scalar, optional
        Also represent this missing label if supplied. Omission selects a
        class-only dtype; explicitly passing `None` includes that value.

    Returns
    -------
    dtype : numpy.dtype
        The promoted dtype if lossless, otherwise `object`.
    """
    arrays = list(vocabularies)
    if missing_label is not _NO_MISSING_LABEL:
        arrays.append(np.asarray([missing_label]))
    dtype = np.result_type(*[array.dtype for array in arrays])
    if dtype.kind == "O":
        return dtype
    for array in arrays:
        if array.dtype == dtype:
            continue
        if array.dtype.kind in "biufc" and dtype.kind in "biufc":
            with np.errstate(invalid="ignore", over="ignore"):
                converted = array.astype(dtype)
                if dtype.kind == "c" and array.dtype.kind != "c":
                    # Real inputs acquire an exactly zero imaginary part.
                    # Discard that part explicitly before casting back, so
                    # this check does not emit ComplexWarning.
                    converted = converted.real
                restored = converted.astype(array.dtype)
            if np.array_equal(
                array, restored, equal_nan=array.dtype.kind in "fc"
            ):
                continue
        elif array.dtype.kind == dtype.kind and dtype.kind in "SU":
            # result_type widens strings of the same kind without truncation.
            continue
        return np.dtype(object)
    return dtype


def _dtype_family(dtype, *, name):
    """Return the label family a NumPy dtype belongs to.

    The dtype of an ordinary label array already determines its family, so
    no value has to be inspected. Only object arrays carry no such evidence
    and are scanned by `_scan_object_labels` instead.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype of a label array.
    name : str
        The name of the checked variable, used in error messages.

    Returns
    -------
    family : "bool" or "int" or "float" or "str"
        The label family of `dtype`.

    Raises
    ------
    TypeError
        If `dtype` is outside the label contract, e.g. complex, bytes, or an
        extended-precision float.
    """
    family = _DTYPE_KIND_FAMILIES.get(dtype.kind)
    if family is None or (dtype.kind in "iuf" and dtype.itemsize > 8):
        raise TypeError(
            f"`{name}` has the unsupported label dtype '{dtype}'. "
            f"{_SUPPORTED_LABELS}"
        )
    return family


def _scalar_label_family(value, *, name):
    """Return the label family of one scalar entry of an object array.

    Parameters
    ----------
    value : object
        One entry of an object array.
    name : str
        The name of the checked variable, used in error messages.

    Returns
    -------
    family : "bool" or "int" or "float" or "str"
        The label family of `value`.

    Raises
    ------
    TypeError
        If `value` is not a supported scalar label. Whether its value is
        storable is decided by `_check_object_values`, once the families of
        the entries are known to agree.
    """
    # Boolean values are their own family, even though `bool` subclasses
    # `int` and `np.bool_` compares equal to `0` and `1`.
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        if np.dtype(type(value)).itemsize > 8:
            raise TypeError(
                f"`{name}` contains the extended-precision float "
                f"'{type(value).__name__}'. {_SUPPORTED_LABELS}"
            )
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        raise TypeError(
            f"`{name}` contains `None`, which is only valid as the configured "
            "`missing_label`. Pass `missing_label=None` to mark these "
            "entries as missing."
        )
    raise TypeError(
        f"`{name}` contains the unsupported scalar label type "
        f"'{type(value).__name__}'. {_SUPPORTED_LABELS}"
    )


def _scan_object_labels(y, missing_label, *, name):
    """Locate the missing labels and the label families of an object array.

    An object dtype carries no information about the values it holds, so its
    entries are the only evidence about both the missing mask and the label
    families. Both are collected in one pass. Entries equal to
    `missing_label` are missing values rather than labels and are therefore
    excluded from the families.

    Parameters
    ----------
    y : numpy.ndarray of dtype object
        Labels including missing values.
    missing_label : scalar or str or None
        The configured missing label, or `_NO_MISSING_LABEL` if the values
        cannot contain one, e.g. a declared class vocabulary.
    name : str
        The name of the checked variable, used in error messages.

    Returns
    -------
    is_missing : numpy.ndarray of shape `y.shape`
        Boolean mask locating `missing_label`.
    families : set of str
        The label families observed among the remaining entries.
    """
    is_missing = []
    families = set()
    for value in y.flat:
        matches = _matches_missing_label(value, missing_label)
        is_missing.append(matches)
        if not matches:
            families.add(_scalar_label_family(value, name=name))
    return np.asarray(is_missing, dtype=bool).reshape(y.shape), families


def _check_object_values(values, families, *, name):
    """Check that the entries of an object array are storable labels.

    Called once the families of the entries are known to agree, so that a
    mixture of numbers and strings is reported as such rather than through
    the value of one of them, e.g. a NaN beside string labels.

    Parameters
    ----------
    values : numpy.ndarray of dtype object
        The observed entries of an object array.
    families : set of str
        The label families found among them.
    name : str
        The name of the checked variable, used in error messages.

    Raises
    ------
    ValueError
        If an integer is beyond 64 bits or a float is nonfinite.
    """
    if "int" in families:
        _check_integer_labels(values, name=name)
    if "float" in families:
        floats = [
            value
            for value in values.flat
            if isinstance(value, (float, np.floating))
        ]
        _check_finite_labels(np.asarray(floats, dtype=float), name=name)


def _combine_families(families, *, name, role):
    """Reduce label families to the one family the values describe.

    The rule follows the role the values play: `_VOCABULARY` and `_LABELS`
    admit one family, `_NUMERICAL_LABELS` admits the integer and floating-point
    families, and `_TASK_AGNOSTIC_LABELS` accepts values valid for either
    classification or regression.

    Parameters
    ----------
    families : set of str
        The label families found among the values.
    name : str
        The name of the checked variable, used in error messages.
    role : "vocabulary" or "labels" or "numerical labels" or \
            "task-agnostic labels"
        The role the values play in the label contract.

    Returns
    -------
    family : "bool" or "int" or "float" or "str" or None
        The common family, or `None` if `families` is empty because no value
        provides evidence about it.

    Raises
    ------
    TypeError
        If the families cannot be combined.
    """
    if not families:
        return None
    if role == _NUMERICAL_LABELS:
        unsupported = families - {"int", "float"}
        if unsupported:
            raise TypeError(
                f"`{name}` must contain numerical labels, "
                f"but contains {_describe_families(unsupported)} values."
            )
        return "float"
    if role == _TASK_AGNOSTIC_LABELS and families <= {"int", "float"}:
        return "float"
    if len(families) == 1:
        return next(iter(families))
    detail = (
        "Boolean values, integers, floating-point values, and strings "
        "are separate families."
    )
    raise TypeError(
        f"`{name}` must contain one label family, but mixes "
        f"{_describe_families(families)} values. {detail}"
    )


def _label_family(values, *, name, role=_VOCABULARY):
    """Return the one label family of a class vocabulary or of labels.

    The representation of `values` decides how they are judged, as described
    by the label and missing-value contract: an ordinary array is judged by
    its dtype, whereas an object array is scanned value by value. Callers
    pass an object array to have short, declared vocabularies checked value
    by value, and the converted label array otherwise.

    Parameters
    ----------
    values : array-like
        Class labels or observed labels, without missing values.
    name : str
        The name of the checked variable, used in error messages.
    role : "vocabulary" or "labels" or "numerical labels", \
            default="vocabulary"
        The role `values` play in the label contract, which decides how
        strictly their families have to agree.

    Returns
    -------
    family : "bool" or "int" or "float" or "str" or None
        The label family, or `None` if `values` is empty and therefore
        provides no evidence about it.

    Raises
    ------
    TypeError
        If the values are outside the label contract or mix families.
    ValueError
        If an integer is beyond 64 bits or a float is nonfinite.
    """
    values = np.asarray(values)
    if values.size == 0:
        return None
    if values.dtype.kind == "O":
        _, families = _scan_object_labels(values, _NO_MISSING_LABEL, name=name)
        # No missing label is in play here, so a nonfinite value is a nonfinite
        # value: it is reported as such whatever the other values are.
        _check_object_values(values, families, name=name)
        return _combine_families(families, name=name, role=role)
    family = _dtype_family(values.dtype, name=name)
    if family == "float":
        _check_finite_labels(values, name=name)
    return _combine_families({family}, name=name, role=role)


def _check_missing_label_value(missing_label):
    """Check that a missing label is a supported scalar value.

    Parameters
    ----------
    missing_label : scalar or str or None
        Value to represent a missing label.

    Raises
    ------
    TypeError
        If `missing_label` is neither a supported real number, nor a string,
        nor `None`.
    ValueError
        If `missing_label` is infinite or an integer beyond 64 bits.
    """
    if missing_label is None or isinstance(missing_label, str):
        return
    if isinstance(missing_label, (bool, np.bool_)):
        raise TypeError(_missing_label_value_message(missing_label))
    if isinstance(missing_label, (int, np.integer)):
        # NumPy integers always fit; a Python integer has arbitrary
        # precision, so its value decides whether it is storable.
        if isinstance(missing_label, int) and not (
            _INT64_MIN <= missing_label <= _UINT64_MAX
        ):
            raise ValueError(
                f"`missing_label={missing_label}` does not fit a 64-bit "
                f"integer dtype. {_STORABLE_INTEGERS}"
            )
        return
    if isinstance(missing_label, (float, np.floating)):
        if np.dtype(type(missing_label)).itemsize > 8:
            raise TypeError(_missing_label_value_message(missing_label))
        if np.isinf(missing_label):
            raise ValueError(
                f"`missing_label={missing_label}` must be finite. NaN marks "
                "a missing label; an infinity is neither a label nor a "
                "missing one."
            )
        return
    raise TypeError(_missing_label_value_message(missing_label))


def _check_missing_label_for_family(
    missing_label, family, *, name=None, description=None
):
    """Check that a missing label is compatible with a label family.

    Encodes the missing-label compatibility of the label contract: `None`
    denotes a missing label in every family, numeric missing labels
    including NaN belong to Boolean, integer, and floating-point labels, and
    string missing labels belong to string labels.

    Parameters
    ----------
    missing_label : scalar or str or None
        Value to represent a missing label.
    family : "bool" or "int" or "float" or "str" or None
        The label family the missing label has to fit, or `None` if no value
        provides evidence about it.
    name : str, default=None
        The name of the variable holding the labels, used in error messages.
    description : str, default=None
        How the labels are named in error messages. Defaults to the
        description of `family`. The roles that merge families pass how
        their labels are named, so that a message names those labels rather
        than the family the compatibility decision was made for.

    Raises
    ------
    TypeError
        If the missing label and the labels belong to different families.
    ValueError
        If `missing_label` is infinite or an integer beyond 64 bits.
    """
    _check_missing_label_value(missing_label)
    if missing_label is None or family is None:
        return
    name = "target object" if name is None else str(name)
    if description is None:
        description = _FAMILY_DESCRIPTIONS[family]
    missing_label_is_string = isinstance(missing_label, str)
    if family == "str":
        if not missing_label_is_string:
            raise TypeError(
                f"`missing_label={missing_label!r}` is a number and is not "
                f"compatible with the string labels in `{name}`. Use a "
                "string missing label or `None` instead."
            )
    elif missing_label_is_string:
        raise TypeError(
            f"`missing_label={missing_label!r}` is a string and is not "
            f"compatible with the {description} labels in `{name}`. Use "
            "`np.nan`, a numeric missing label, or `None` instead."
        )


def _check_compatible_kinds(observed, declared, *, name):
    """Check that observed labels and declared classes share one label kind.

    A comparison of labels and classes of different kinds cannot report
    which class a label missed, because none of them could ever match: a
    string label is not a numeric category however it is spelled. Widths and
    numeric families may differ, so the observed `0.0` of a declared integer
    class `0`, or of a declared `False`, remains that class.

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
        If the labels and the classes belong to different label kinds.
    """
    observed_kind = _label_kind(np.asarray(observed), name="y")
    declared_kind = _label_kind(
        np.asarray(list(declared), dtype=object), name=name
    )
    if observed_kind is not None and observed_kind != declared_kind:
        raise TypeError(
            f"The labels in `y` are not type-compatible with {name}."
        )


def _label_kind(values, *, name):
    """Return the coarse kind observed labels and declared classes share.

    Integer and floating-point labels describe the same categories, e.g. an
    observed `0.0` matches a declared integer class `0`, so labels and
    classes only have to agree on their kind rather than on their family.

    Parameters
    ----------
    values : array-like
        Class labels or observed labels, without missing values.
    name : str
        The name of the checked variable, used in error messages.

    Returns
    -------
    kind : "bool" or "numeric" or "str" or None
        The label kind, or `None` if `values` provides no evidence about it.
    """
    family = _label_family(values, name=name, role=_LABELS)
    return None if family is None else _FAMILY_KINDS[family]


def _target_type_family(target_type):
    """Map the dtype or scalar type of labels to their label family.

    `check_missing_label` accepts the dtype or Python type of the labels
    rather than their family, so that its public signature stays unchanged.
    Only the missing label is judged here, so a dtype outside the label
    contract carries no family instead of being rejected, just as an object
    dtype does; the label helpers reject it when they see the values.

    Parameters
    ----------
    target_type : numpy.dtype or type or tuple
        The dtype or scalar type of the labels a missing label has to fit.

    Returns
    -------
    family : "bool" or "int" or "float" or "str" or None
        The label family, or `None` if `target_type` constrains no missing
        label.
    """
    try:
        dtype = np.dtype(target_type)
    except TypeError:
        return None
    if dtype.kind == "S":
        return "str"
    return _DTYPE_KIND_FAMILIES.get(dtype.kind)


def _missing_mask_and_family(y, missing_label, *, name, role=_LABELS):
    """Locate missing labels and check the family of the observed ones.

    This is the one place where the label and missing-value contract is
    enforced for observed labels: which values are labels at all, which
    value denotes a missing one, and which values remain as evidence about
    the label family.

    An ordinary array announces the family of its observed entries through
    its dtype. Its mask is built first because an entirely missing array
    provides no family evidence. An object array announces nothing, so its
    mask and its families are collected in one scan.

    Parameters
    ----------
    y : numpy.ndarray
        Labels including missing values.
    missing_label : scalar or str or None
        Value to represent a missing label.
    name : str
        The name of the checked variable, used in error messages.
    role : "labels" or "numerical labels" or "task-agnostic labels", \
            default="labels"
        The role the values of `y` play in the label contract.

    Returns
    -------
    is_missing : numpy.ndarray of shape `y.shape`
        Boolean mask locating `missing_label` in `y`.
    family : "bool" or "int" or "float" or "str" or None
        The label family of the observed labels, or `None` if `y` holds no
        observed label.

    Raises
    ------
    TypeError
        If the labels are outside the label contract, mix families, or are
        incompatible with `missing_label`.
    ValueError
        If an observed integer is beyond 64 bits or an observed float is
        nonfinite.
    """
    _check_missing_label_value(missing_label)
    if y.dtype.kind == "O":
        is_missing, families = _scan_object_labels(y, missing_label, name=name)
        _check_object_values(y[~is_missing], families, name=name)
        family = _combine_families(families, name=name, role=role)
        _check_missing_label_for_family(
            missing_label,
            family,
            name=name,
            description=_describe_observed_families(families),
        )
        return is_missing, family

    is_missing = _typed_missing_mask(y, missing_label)
    if is_missing.all():
        _check_missing_label_for_family(missing_label, None, name=name)
        return is_missing, None

    observed_family = _dtype_family(y.dtype, name=name)
    family = _combine_families({observed_family}, name=name, role=role)
    _check_missing_label_for_family(
        missing_label,
        family,
        name=name,
        description=_FAMILY_DESCRIPTIONS[observed_family],
    )
    if y.dtype.kind == "f":
        _check_finite_labels(y[~is_missing], name=name)
    return is_missing, family


def _typed_missing_mask(y, missing_label):
    """Locate a missing label in an array without coercing its values.

    A missing label of another family than the labels cannot occur among
    them, so the comparison is skipped rather than performed in a dtype that
    both values are forced into. Numeric missing labels use a lossless
    comparison dtype, so that neighboring large integer labels are not
    merged onto the missing label.

    Parameters
    ----------
    y : numpy.ndarray
        Labels of a non-object dtype, including missing values.
    missing_label : scalar or str or None
        Value to represent a missing label.

    Returns
    -------
    is_missing : numpy.ndarray of shape `y.shape`
        Boolean mask locating `missing_label` in `y`.
    """
    if missing_label is None:
        return np.zeros(y.shape, dtype=bool)
    if _is_nan_missing_label(missing_label):
        if y.dtype.kind == "f":
            return np.isnan(y)
        return np.zeros(y.shape, dtype=bool)
    if isinstance(missing_label, str) != (y.dtype.kind == "U"):
        return np.zeros(y.shape, dtype=bool)
    if isinstance(missing_label, str):
        return y == missing_label
    comparison_dtype = _lossless_decode_dtype([y], missing_label)
    return y.astype(comparison_dtype, copy=False) == missing_label


def _matches_missing_label(value, missing_label):
    """Check whether one scalar entry of an object array is missing.

    Parameters
    ----------
    value : object
        One entry of an object array, not necessarily a supported label.
    missing_label : scalar or str or None
        The configured missing label, or `_NO_MISSING_LABEL` if the values
        cannot contain one.

    Returns
    -------
    matches : bool
        Whether `value` is the configured missing label.
    """
    if missing_label is _NO_MISSING_LABEL:
        return False
    if missing_label is None:
        return value is None
    if _is_nan_missing_label(missing_label):
        return isinstance(value, (float, np.floating)) and bool(
            np.isnan(value)
        )
    if isinstance(missing_label, str):
        return isinstance(value, str) and value == missing_label
    if not isinstance(
        value, (bool, np.bool_, int, np.integer, float, np.floating)
    ):
        return False
    # Object arrays retain NumPy scalars, whose equality promotes integers to
    # floats and can thereby merge a large label onto the missing label. Python
    # scalars compare exactly across integers and floats.
    return _python_scalar(value) == _python_scalar(missing_label)


def _holds_missing_label(values, missing_label):
    """Check whether an array can store the missing label without loss.

    A dtype able to hold the labels need not be able to hold their missing
    label, e.g. integer labels cannot store `np.nan` and a `float64` array
    cannot store `None`. Components writing the missing label into an
    existing label array require that it can.

    Parameters
    ----------
    values : numpy.ndarray
        The label array the missing label would be stored in.
    missing_label : scalar or str or None
        Value to represent a missing label.

    Returns
    -------
    holds_missing_label : bool
        Whether `values.dtype` represents both the labels and the missing
        label.
    """
    return _lossless_decode_dtype([values], missing_label) == values.dtype


def _missing_label_kind(missing_label):
    """Return the label kind a missing label belongs to.

    `None` denotes a missing label in every family and therefore belongs to
    no kind of its own.

    Parameters
    ----------
    missing_label : scalar or str or None
        Value to represent a missing label.

    Returns
    -------
    kind : "numeric" or "str" or None
        The kind of labels `missing_label` can mark a missing one of, or
        `None` if it can mark one of every kind.
    """
    if missing_label is None:
        return None
    return "str" if isinstance(missing_label, str) else "numeric"


def _is_nan_missing_label(missing_label):
    """Return whether a missing label is a floating-point NaN."""
    return isinstance(missing_label, (float, np.floating)) and bool(
        np.isnan(missing_label)
    )


def _python_scalar(value):
    """Return the Python scalar equivalent of a NumPy scalar."""
    return value.item() if isinstance(value, np.generic) else value


def _check_integer_labels(values, *, name):
    """Check that the integers among object entries fit a 64-bit dtype.

    Python integers have arbitrary precision, so their values decide whether
    they are storable. Individually storable integers can still lack a
    *common* integer dtype, e.g. `-1` beside `2**64 - 1`. NumPy would then
    promote them to `float64` and lose the identity of the large ones.

    Parameters
    ----------
    values : numpy.ndarray of dtype object
        The observed entries of an object array.
    name : str
        The name of the checked variable, used in error messages.

    Raises
    ------
    ValueError
        If no signed or unsigned 64-bit integer dtype holds every integer.
    """
    integers = [
        _python_scalar(value)
        for value in values.flat
        if isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    ]
    if not integers:
        return
    smallest, largest = min(integers), max(integers)
    if smallest < _INT64_MIN or largest > _UINT64_MAX:
        outside = smallest if smallest < _INT64_MIN else largest
        raise ValueError(
            f"`{name}` contains the integer {outside}, which does not fit a "
            f"64-bit integer dtype. {_STORABLE_INTEGERS}"
        )
    if smallest < 0 and largest > _INT64_MAX:
        raise ValueError(
            f"`{name}` contains the integers {smallest} and {largest}, which "
            f"do not fit one 64-bit integer dtype. {_STORABLE_INTEGERS}"
        )


def _check_finite_labels(values, *, name):
    """Check that floating-point labels are finite.

    Parameters
    ----------
    values : numpy.ndarray
        Floating-point labels without missing values.
    name : str
        The name of the checked variable, used in error messages.

    Raises
    ------
    ValueError
        If any value is NaN or infinite.
    """
    is_finite = np.isfinite(values)
    if is_finite.all():
        return
    if np.isnan(values[~is_finite]).any():
        raise ValueError(
            f"`{name}` contains NaN, which is only valid as the configured "
            "`missing_label`. Pass `missing_label=np.nan` to mark these "
            "entries as missing."
        )
    raise ValueError(
        f"`{name}` contains an infinite value, which is neither a label nor "
        "a missing label."
    )


def _describe_observed_families(families):
    """Name the label families found among observed labels.

    The numerical and task-agnostic roles merge integers and floating-point
    values into one family, because both describe a numerical label. An
    error message names the labels rather than that merged family, so that
    integer labels are not reported as floating-point ones.

    Parameters
    ----------
    families : set of str
        The label families found among the observed labels.

    Returns
    -------
    description : str or None
        How the observed labels are named in an error message, or `None` if
        `families` is empty, because no observed label is then there to be
        incompatible with a missing label.
    """
    if not families:
        return None
    if families == {"int", "float"}:
        return "numerical"
    return _describe_families(families)


def _describe_families(families):
    """Name label families in a stable order for an error message."""
    described = [
        _FAMILY_DESCRIPTIONS[family]
        for family in _FAMILY_ORDER
        if family in families
    ]
    if len(described) == 1:
        return described[0]
    return ", ".join(described[:-1]) + f" and {described[-1]}"


def _missing_label_value_message(missing_label):
    """Describe an unsupported missing label."""
    return (
        f"`missing_label` has type '{type(missing_label).__name__}', but "
        "must be a number of at most 64 bits, a string, `np.nan`, or `None`."
    )
