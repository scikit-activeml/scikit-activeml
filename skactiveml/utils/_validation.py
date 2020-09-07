import numpy as np

from collections.abc import Iterable
from sklearn.utils import check_array, check_scalar


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

    Returns
    -------
    missing_label : number | str | None | np.nan
        Input missing label if no error has been raised.
    """
    is_None = missing_label is None
    is_character = np.issubdtype(type(missing_label), np.character)
    is_number = np.issubdtype(type(missing_label), np.number)
    if not is_number and not is_character and not is_None:
        raise TypeError(
            "'missing_label' has type '{}', but must be a either a number, "
            "a string, np.nan, or None.".format(type(missing_label)))
    if target_type is not None:
        is_object_type = np.issubdtype(target_type, np.object)
        is_character_type = np.issubdtype(target_type, np.character)
        is_number_type = np.issubdtype(target_type, np.number)
        if (is_character_type and is_number) or (
                is_number_type and is_character) or (
                is_object_type and not is_None):
            name = 'target object' if name is None else str(name)
            raise TypeError(
                "'missing_label' has type '{}' and is not compatible to the "
                "type '{}' of '{}'.".format(
                    type(missing_label), target_type, name))

    return missing_label


def check_classes(classes):
    """Check whether class labels uniformly strings or numbers.

    Parameters
    ----------
    classes : array-like, shape (n_classes)
        Array of class labels.

    Returns
    -------
    classes : array-like, shape (n_classes)
        Sorted array of class labels.
    """
    if not isinstance(classes, Iterable):
        raise TypeError(
            "'classes' is not iterable. Got {}".format(type(classes)))
    try:
        classes = sorted(set(classes))
        return classes
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in classes))
        raise TypeError(
            "'classes' must be uniformly strings or numbers. Got {}".format(
                types))


def check_cost_matrix(cost_matrix, n_classes):
    """Check whether class labels uniformly strings or numbers.

    Parameters
    ----------
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix.
    n_classes : int
        Number of classes.

    Returns
    -------
    cost_matrix : numpy.ndarray, shape (n_classes, n_classes)
        Cost matrix.
    """
    check_scalar(n_classes, target_type=int, name='n_classes', min_val=1)
    cost_matrix = check_array(np.asarray(cost_matrix, dtype=float),
                              ensure_2d=True)
    if cost_matrix.shape != (n_classes, n_classes):
        raise ValueError(
            "'cost_matrix' must have shape ({}, {}). "
            "Got {}.".format(n_classes, n_classes, cost_matrix.shape))
    return cost_matrix
