import numpy as np
from iteration_utilities import deepflatten

# Define constant for missing label used throughout the package.
MISSING_LABEL = np.nan


def is_unlabeled(y, missing_label=MISSING_LABEL):
    """Creates a boolean mask indicating missing labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to missing labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    is_unlabeled : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating missing labels in y.
    """
    check_missing_label(missing_label)
    if len(y) == 0:
        return np.array(y, dtype=bool)
    if not isinstance(y, np.ndarray):
        types = set(
            t.__qualname__ for t in set(type(v) for v in deepflatten(y))
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
    target_type = np.append(y.ravel(), missing_label).dtype
    check_missing_label(missing_label, target_type=target_type, name="y")
    if (y.ndim == 2 and np.size(y, axis=1) == 0) or y.ndim > 2:
        raise ValueError(
            "'y' must be of shape (n_samples) or '(n_samples, "
            "n_features)' with 'n_samples > 0' and "
            "'n_features > 0'."
        )
    if missing_label is np.nan:
        return np.isnan(y)
    else:
        return y == missing_label


def is_labeled(y, missing_label=MISSING_LABEL):
    """Creates a boolean mask indicating present labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    is_unlabeled : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating present labels in y.
    """
    return ~is_unlabeled(y, missing_label)


def unlabeled_indices(y, missing_label=MISSING_LABEL):
    """Return an array of indices indicating missing labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    unlbld_indices : numpy.ndarray, shape (n_samples) or (n_samples, 2)
        Index array of missing labels. If y is a 2D-array, the indices
        have shape `(n_samples, 2), otherwise it has the shape `(n_samples)`.
    """
    is_unlbld = is_unlabeled(y, missing_label)
    unlbld_indices = np.argwhere(is_unlbld)
    return unlbld_indices[:, 0] if is_unlbld.ndim == 1 else unlbld_indices


def labeled_indices(y, missing_label=MISSING_LABEL):
    """Return an array of indices indicating present labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    lbld_indices : numpy.ndarray, shape (n_samples) or (n_samples, 2)
        Index array of present labels. If y is a 2D-array, the indices
        have shape `(n_samples, 2), otherwise it has the shape `(n_samples)`.
    """
    is_lbld = is_labeled(y, missing_label)
    lbld_indices = np.argwhere(is_lbld)
    return lbld_indices[:, 0] if is_lbld.ndim == 1 else lbld_indices


def check_missing_label(missing_label, target_type=None, name=None):
    """Check whether a missing label is compatible to a given target type.

    Parameters
    ----------
    missing_label : number | str | None | np.nan
        Symbol to represent a missing label.
    target_type : type or tuple
        Acceptable data types for the parameter 'missing_label'.
    name : str
        The name of the variable to which 'missing_label' is not compatible.
        The name will be printed in error messages.
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
    missing_label1 : number | str | None | np.nan
        Symbol to represent a missing label.
    missing_label2 : number | str | None | np.nan
        Other symbol to represent a missing label.

    Raises
    -------
    ValueError
        If the parameter's value violates the given bounds.
    """
    if not is_unlabeled([missing_label1], missing_label=missing_label2)[0]:
        raise ValueError(
            f"missing_label1={missing_label1} and "
            f"missing_label2={missing_label2} must be equal."
        )
