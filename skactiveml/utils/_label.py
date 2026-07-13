import numpy as np

# Define constant for missing label used throughout the package.

MISSING_LABEL = np.nan


def _deepflatten(to_flatten):
    """Flattens the iterable `to_flatten` recursively, in such a way that only
    elementary items are returned in an one-dimensional list.

    Parameters
    ----------
    to_flatten : Iterable
        The iterable to flatten.

    Returns
    -------
    flattened_list : list
        A list that contains all elements of `to_flatten` without being nested.
    """
    # list to keep track of objects to flatten
    iterables = [to_flatten]
    # list to save all non-iterable elements
    flattened_list = []

    while iterables:
        # remove last element and iterate over it
        current_iterable = iterables.pop()
        for e in current_iterable:
            # check objects that return themselves as iterable (e.g. when
            # iterating over strings)
            if e == current_iterable:
                flattened_list.append(e)
            # if iterable, iterate over its elements
            elif hasattr(e, "__iter__"):
                iterables.append(e)
            # if non-iterable element, append to flattened list
            else:
                flattened_list.append(e)
    return flattened_list


def is_unlabeled(
    y,
    missing_label=MISSING_LABEL,
    *,
    target_type="single-output",
):
    """Creates a boolean mask indicating missing labels.

    Parameters
    ----------
    y : array-like of shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to missing labels.
    missing_label : number or str or None or np.nan, default=np.nan
        Value to represent a missing label.
    target_type : {"single-output", "multi-label"}, default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must be either contain
        only observed labels or only `missing_label` values, i.e., no mixing
        within a row.

    Returns
    -------
    is_unlbld : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Boolean mask indicating missing labels in `y`.

        - If `target_type="single-output"`, `is_unlbld` has the same shape as
          `y`.
        - If `target_type="multi-label"`, `is_unlbld` is of shape
          `(n_samples,)`.
    """
    check_missing_label(missing_label)
    if target_type == "auto":
        raise ValueError(
            "`target_type='auto'` is not supported by label helpers; pass a "
            "resolved target type."
        )
    if target_type not in {"single-output", "multi-label"}:
        raise ValueError(
            "`target_type` must be either 'single-output' or 'multi-label'."
        )
    if len(y) == 0:
        y = np.asarray(y)
        if target_type == "multi-label":
            if y.ndim != 2:
                raise ValueError(
                    "`y` must be two-dimensional when "
                    "`target_type='multi-label'`."
                )
            return np.zeros(y.shape[0], dtype=bool)
        return np.array(y, dtype=bool)
    if not isinstance(y, np.ndarray):
        types = set(
            t.__qualname__ for t in set(type(v) for v in _deepflatten(y))
        )
        types.add(type(missing_label).__qualname__)
        is_number = False
        is_character = False
        for t in types:
            t = object if t == "NoneType" else t
            is_character = (
                True if np.issubdtype(t, np.character) else is_character
            )
            is_number = True if np.issubdtype(t, np.number) else is_number
            if is_character and is_number:
                raise TypeError(
                    "'y' must be uniformly strings or numbers. "
                    "'NoneType' is allowed. Got {}".format(types)
                )
        y = np.asarray(y)
    y_dtype = np.result_type(y, np.asarray(missing_label))
    check_missing_label(missing_label, target_type=y_dtype, name="y")

    # Check requirements for labels `y`.
    if y.ndim not in (1, 2):
        raise ValueError(
            "`y` must have shape (n_samples,) or (n_samples, n_outputs)."
        )
    if y.shape[0] == 0 or (y.ndim == 2 and y.shape[1] == 0):
        raise ValueError(
            "`y` must have `n_samples > 0` and (if two-dimensional) "
            "`n_outputs > 0`."
        )

    if target_type == "multi-label" and y.ndim != 2:
        raise ValueError(
            "`y` must be two-dimensional when `target_type='multi-label'`."
        )

    # Compute elementwise missing mask.
    missing_is_nan = isinstance(missing_label, float) and np.isnan(
        missing_label
    )
    if missing_is_nan:
        is_missing = np.isnan(y.astype(float))
    else:
        y = y.astype(y_dtype)
        is_missing = y == missing_label

    # Handle single output.
    if target_type == "single-output":
        return is_missing

    # Handle multiple outputs.
    row_any = is_missing.any(axis=1)
    row_all = is_missing.all(axis=1)
    mixed_rows = row_any ^ row_all
    if mixed_rows.any():
        raise ValueError(
            "Each row `y[i]` must contain either only observed labels or only "
            "`missing_label` values (no mixing within a row)."
        )

    return row_all


def is_labeled(
    y,
    missing_label=MISSING_LABEL,
    *,
    target_type="single-output",
):
    """Creates a boolean mask indicating present labels.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number or str or None or np.nan, default=np.nan
        Value to represent a missing label.
    target_type : {"single-output", "multi-label"}, default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must be either contain
        only observed labels or only `missing_label` values, i.e., no mixing
        within a row.

    Returns
    -------
    is_lbld : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Boolean mask indicating present labels in `y`.

        - If `target_type="single-output"`, `is_lbld` has the same shape as
          `y`.
        - If `target_type="multi-label"`, `is_lbld` has shape `(n_samples,)`.
    """
    return ~is_unlabeled(
        y=y,
        missing_label=missing_label,
        target_type=target_type,
    )


def unlabeled_indices(
    y,
    missing_label=MISSING_LABEL,
    *,
    target_type="single-output",
):
    """Return an array of indices indicating missing labels.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number or str or None or np.nan, default=np.nan
        Value to represent a missing label.
    target_type : {"single-output", "multi-label"}, default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must be either contain
        only observed labels or only `missing_label` values, i.e., no mixing
        within a row.

    Returns
    -------
    unlbld_indices : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
        Index array of missing labels.

        - If `target_type="single-output"` and `y` is a 2D-array,
          `unlbld_indices`
          has the shape `(n_samples, 2)`.
        - Otherwise, `unlbld_indices` has the shape `(n_samples,)`.
    """
    is_unlbld = is_unlabeled(
        y=y,
        missing_label=missing_label,
        target_type=target_type,
    )
    unlbld_indices = np.argwhere(is_unlbld)
    return unlbld_indices[:, 0] if is_unlbld.ndim == 1 else unlbld_indices


def labeled_indices(
    y,
    missing_label=MISSING_LABEL,
    *,
    target_type="single-output",
):
    """Return an array of indices indicating present labels.

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number or str or None or np.nan, default=np.nan
        Value to represent a missing label.
    target_type : {"single-output", "multi-label"}, default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must be either contain
        only observed labels or only `missing_label` values, i.e., no mixing
        within a row.

    Returns
    -------
    lbld_indices : numpy.ndarray of shape (n_samples) or (n_samples, 2)
        Index array of present labels.

        - If `target_type="single-output"` and `y` is a 2D-array,
          `lbld_indices`
          has the shape `(n_samples, 2)`.
        - Otherwise, `lbld_indices` has the shape `(n_samples,)`.
    """
    is_lbld = is_labeled(
        y,
        missing_label,
        target_type=target_type,
    )
    lbld_indices = np.argwhere(is_lbld)
    return lbld_indices[:, 0] if is_lbld.ndim == 1 else lbld_indices


def check_missing_label(missing_label, target_type=None, name=None):
    """Check whether a missing label is compatible to a given target type.

    Parameters
    ----------
    missing_label : number or str or None or np.nan
        Value to represent a missing label.
    target_type : Type or tuple, default=None
        Acceptable data types for the parameter `missing_label` if it is not
        set to None.
    name : str, default=None
        The name of the variable to which `missing_label` is not compatible.
        The name will be printed in error messages if it is not None.
    """
    is_None = missing_label is None
    is_character = np.issubdtype(type(missing_label), np.character)
    is_number = np.issubdtype(type(missing_label), np.number)
    if not is_number and not is_character and not is_None:
        raise TypeError(
            "'missing_label' has type '{}', but must be a either a number, "
            "a string, np.nan, or None.".format(type(missing_label))
        )
    if target_type is not None:
        is_object_type = np.issubdtype(target_type, np.object_)
        is_character_type = np.issubdtype(target_type, np.character)
        is_number_type = np.issubdtype(target_type, np.number)
        if (
            (is_character_type and is_number)
            or (is_number_type and is_character)
            or (is_object_type and not is_None)
        ):
            name = "target object" if name is None else str(name)
            raise TypeError(
                "'missing_label' has type '{}' and is not compatible to the "
                "type '{}' of '{}'.".format(
                    type(missing_label), target_type, name
                )
            )


def check_equal_missing_label(missing_label1, missing_label2):
    """Check whether two missing label values are equal to each other.

    Parameters
    ----------
    missing_label1 : number or str or None or np.nan
        Value to represent a missing label.
    missing_label2 : number or str or None or np.nan
        Other value to represent a missing label.

    Raises
    -------
    ValueError
        If the two missing labels are not equal.
    """
    if not is_unlabeled([missing_label1], missing_label=missing_label2)[0]:
        raise ValueError(
            f"missing_label1={missing_label1} and "
            f"missing_label2={missing_label2} must be equal."
        )
