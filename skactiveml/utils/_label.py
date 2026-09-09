import numpy as np

from ._label_dtype import (
    _LABELS,
    _NUMERICAL_LABELS,
    _TASK_AGNOSTIC_LABELS,
    _as_label_array,
    _check_missing_label_for_family,
    _check_missing_label_value,
    _missing_label_kind,
    _matches_missing_label,
    _missing_mask_and_family,
    _target_type_family,
)

# Define constant for missing label used throughout the package.

MISSING_LABEL = np.nan


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
    target_type : "single-output" or "multi-label", default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must contain either
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
    role = _LABELS if target_type == "multi-label" else _TASK_AGNOSTIC_LABELS
    return _is_unlabeled_with_role(
        y,
        missing_label=missing_label,
        target_type=target_type,
        role=role,
    )


def _is_unlabeled_with_role(
    y,
    missing_label=MISSING_LABEL,
    *,
    target_type="single-output",
    role,
):
    """Return the missing mask after role-specific label validation."""
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
    y = _as_label_array(y)

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

    is_missing, _ = _missing_mask_and_family(
        y,
        missing_label,
        name="y",
        role=role,
    )

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
    target_type : "single-output" or "multi-label", default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must contain either
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


def _check_labels(
    y,
    missing_label=MISSING_LABEL,
    *,
    target_type="single-output",
    task="classification",
):
    """Check labels against the label and missing-value contract.

    Components that only have to establish that `y` is admissible, without
    using the resulting mask, call this instead of discarding the mask of
    `is_unlabeled`.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Class labels or numerical labels, including values equal to
        `missing_label`.
    missing_label : number or str or None or np.nan, default=np.nan
        Value to represent a missing label.
    target_type : "single-output" or "multi-label", default="single-output"
        The resolved target type.
    task : "classification" or "regression", default="classification"
        The prediction task the labels describe. Numerical labels may mix
        integer and floating-point values but are neither categories nor
        strings.

    Raises
    ------
    TypeError
        If the labels are outside the label contract, mix label kinds, or
        are incompatible with `missing_label`.
    ValueError
        If an observed label is nonfinite or an integer beyond 64 bits.
    """
    if task not in {"classification", "regression"}:
        raise ValueError(
            "`task` must be either 'classification' or 'regression'."
        )
    _is_unlabeled_with_role(
        y,
        missing_label=missing_label,
        target_type=target_type,
        role=_NUMERICAL_LABELS if task == "regression" else _LABELS,
    )


def _observed_numerical_labels(y, missing_label):
    """Return the observed numerical labels as floating-point values.

    Regression labels are stored as an object array whenever the missing
    label and the labels have no common numeric dtype, e.g. for
    `missing_label=None`. Object entries do not support the arithmetic a
    regressor performs on them, so the observed labels are provided as an
    ordinary floating-point view alongside the mask locating them.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Numerical labels including missing values.
    missing_label : number or str or None or np.nan
        Value to represent a missing label.

    Returns
    -------
    is_lbld : numpy.ndarray of shape (n_samples,)
        Boolean mask indicating observed labels in `y`.
    y_observed : numpy.ndarray of shape (n_observed,)
        The observed labels as `float64` values.
    """
    is_lbld = ~_is_unlabeled_with_role(
        y,
        missing_label,
        role=_NUMERICAL_LABELS,
    )
    observed = _as_label_array(y)[is_lbld]
    return is_lbld, np.asarray(observed, dtype=float)


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
    target_type : "single-output" or "multi-label", default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must contain either
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
    target_type : "single-output" or "multi-label", default="single-output"
        The resolved target type. For multi-label targets, `y` must be
        two-dimensional. Furthermore, a row `y[i]` must contain either
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
    target_type : numpy.dtype or Type or tuple, default=None
        The dtype or scalar type of the labels `missing_label` has to fit. If
        `None`, only `missing_label` itself is checked. A dtype carrying no
        label family, such as `object`, constrains no missing label; the
        label-mask functions then check the missing label against the values
        they see.
    name : str, default=None
        The name of the variable to which `missing_label` is not compatible.
        The name will be printed in error messages if it is not None.

    Raises
    ------
    TypeError
        If `missing_label` is no supported missing label, or belongs to another
        label family than `target_type`.
    ValueError
        If `missing_label` is infinite or an integer beyond 64 bits.
    """
    if target_type is None:
        _check_missing_label_value(missing_label)
        return
    _check_missing_label_for_family(
        missing_label, _target_type_family(target_type), name=name
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
    TypeError
        If the two values mark missing labels of different kinds, e.g.
        numeric `np.nan` and the string `"nan"`.
    ValueError
        If the two missing labels are of one kind but are not equal.
    """
    _check_missing_label_value(missing_label1)
    _check_missing_label_value(missing_label2)
    if _matches_missing_label(missing_label1, missing_label2):
        return
    message = (
        f"missing_label1={missing_label1} and "
        f"missing_label2={missing_label2} must be equal."
    )
    kind1 = _missing_label_kind(missing_label1)
    kind2 = _missing_label_kind(missing_label2)
    if kind1 is not None and kind2 is not None and kind1 != kind2:
        raise TypeError(message)
    raise ValueError(message)
