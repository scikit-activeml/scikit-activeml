from collections.abc import Iterable

import numpy as np
import sklearn
from sklearn.utils.validation import check_array, column_or_1d, \
    assert_all_finite, check_consistent_length

# Define constant for missing label used throughout the package.
MISSING_LABEL = np.nan


def check_scalar(x, name, target_type, min_inclusive=True, max_inclusive=True,
                 min_val=None, max_val=None):
    """Validate scalar parameters type and value.

    Parameters
    ----------
    x : object
        The scalar parameter to validate.
    name : str
        The name of the parameter to be printed in error messages.
    target_type : type or tuple
        Acceptable data types for the parameter.
    min_val : float or int, optional (default=None)
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.
    min_inclusive : bool, optional (default=True)
        If true, the minimum valid value is inclusive, otherwise exclusive.
    max_val : float or int, optional (default=None)
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.
    max_inclusive : bool, optional (default=True)
        If true, the maximum valid value is inclusive, otherwise exclusive.

    Raises
    -------
    TypeError
        If the parameter's type does not match the desired type.
    ValueError
        If the parameter's value violates the given bounds.
    """
    if not isinstance(x, target_type):
        raise TypeError('`{}` must be an instance of {}, not {}.'
                        .format(name, target_type, type(x)))
    if min_inclusive:
        if min_val is not None and x < min_val:
            raise ValueError('`{}`= {}, must be >= '
                             '{}.'.format(name, x, min_val))
    else:
        if min_val is not None and x <= min_val:
            raise ValueError('`{}`= {}, must be > '
                             '{}.'.format(name, x, min_val))

    if max_inclusive:
        if max_val is not None and x > max_val:
            raise ValueError('`{}`= {}, must be <= '
                             '{}.'.format(name, x, max_val))
    else:
        if max_val is not None and x >= max_val:
            raise ValueError('`{}`= {}, must be < '
                             '{}.'.format(name, x, max_val))


def check_classifier_params(classes, missing_label, cost_matrix=None):
    """Check whether the parameters are compatible to each other (only if
    `classes` is not None).

    Parameters
    ----------
    classes : array-like, shape (n_classes)
        Array of class labels.
    missing_label : {number, str, None, np.nan}
        Symbol to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes), default=None
        Cost matrix. If None, cost matrix will be not checked.
    """
    check_missing_label(missing_label)
    if classes is not None:
        check_classes(classes)
        dtype = np.append(classes, missing_label).dtype
        check_missing_label(missing_label, target_type=dtype, name='classes')
        if cost_matrix is not None:
            check_cost_matrix(cost_matrix=cost_matrix, n_classes=len(classes))
    else:
        if cost_matrix is not None:
            raise ValueError("You cannot specify 'cost_matrix' without "
                             "specifying 'classes'.")


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
            "a string, np.nan, or None.".format(type(missing_label)))
    if target_type is not None:
        is_object_type = np.issubdtype(target_type, np.object_)
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


def check_classes(classes):
    """Check whether class labels are uniformly strings or numbers.

    Parameters
    ----------
    classes : array-like, shape (n_classes)
        Array of class labels.
    """
    if not isinstance(classes, Iterable):
        raise TypeError(
            "'classes' is not iterable. Got {}".format(type(classes)))
    try:
        classes_sorted = np.array(sorted(set(classes)))
        if len(classes) != len(classes_sorted):
            raise ValueError("Duplicate entries in 'classes'.")
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in classes))
        raise TypeError(
            "'classes' must be uniformly strings or numbers. Got {}".format(
                types))


def check_cost_matrix(cost_matrix, n_classes, only_non_negative=True,
                      contains_non_zero=True, diagonal_is_zero=True):
    """Check whether cost matrix has shape `(n_classes, n_classes)`.

    Parameters
    ----------
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix.
    n_classes : int
        Number of classes.
    only_non_negative : bool, optional (default=True)
        This parameter determines whether the matrix must contain only non
        negative cost entries.
    contains_non_zero : bool, optional (default=True)
        This parameter determines whether the matrix must contain at least on
        non-zero cost entry.
    diagonal_is_zero : bool, optional (default=True)
        This parameter determines whether the diagonal cost entries must be
        zero.

    Returns
    -------
    cost_matrix_new : np.ndarray, shape (n_classes, n_classes)
        Numpy array as cost matrix.
    """
    check_scalar(n_classes, target_type=int, name='n_classes', min_val=1)
    cost_matrix_new = check_array(np.array(cost_matrix, dtype=float),
                                  ensure_2d=True)
    if cost_matrix_new.shape != (n_classes, n_classes):
        raise ValueError(
            "'cost_matrix' must have shape ({}, {}). "
            "Got {}.".format(n_classes, n_classes, cost_matrix_new.shape))
    if only_non_negative and np.sum(cost_matrix_new < 0) > 0:
        raise ValueError(
            "'cost_matrix' must contain only non-negative cost entries."
        )
    if n_classes != 1 and \
            contains_non_zero and (np.sum(cost_matrix_new != 0) == 0):
        raise ValueError(
            "'cost_matrix' must contain at least one non-zero cost entry."
        )
    if diagonal_is_zero and np.sum(np.diag(cost_matrix_new)) > 0:
        raise ValueError(
            "'cost_matrix' must contain at only zero cost entries on its "
            "diagonal."
        )
    return cost_matrix_new


def check_X_y(X, y, X_cand=None, sample_weight=None, sample_weight_cand=None,
              accept_sparse=False, *, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              allow_nan=None, ensure_min_samples=1, ensure_min_features=1,
              y_numeric=False, estimator=None, missing_label=MISSING_LABEL):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Labeled input data.

    y : nd-array, list or sparse matrix
        Labels for X.

    X_cand : nd-array, list or sparse matrix (default=None)
        Unlabeled input data

    sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights.

    sample_weight_cand : array-like of shape (n_candidates,) (default=None)
            Sample weights of the candidates.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    allow_nan : boolean (default=None)
        Whether to allow np.nan in y.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    missing_label : {scalar, string, np.nan, None}, (default=np.nan)
        Value to represent a missing label.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.

    X_cand : object
        The converted and validated X_cand
        Only returned if X_cand is not None.

    sample_weight : np.ndarray
        The converted and validated sample_weight.

    sample_weight_cand : np.ndarray
        The converted and validated sample_weight_cand.
        Only returned if X_cand is not None.
    """
    if y is None:
        raise ValueError("y cannot be None")

    if allow_nan is None:
        allow_nan = True if missing_label is np.nan else False

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, accept_sparse='csr', force_all_finite=True,
                        ensure_2d=False, dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        assert_all_finite(y, allow_nan=allow_nan)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)
    check_consistent_length(X, y)

    if X_cand is not None:
        X_cand = check_array(X_cand, accept_sparse=accept_sparse,
                             accept_large_sparse=accept_large_sparse,
                             dtype=dtype, order=order, copy=copy,
                             force_all_finite=force_all_finite,
                             ensure_2d=ensure_2d, allow_nd=allow_nd,
                             ensure_min_samples=ensure_min_samples,
                             ensure_min_features=ensure_min_features,
                             estimator=estimator)
        if X_cand.shape[1] is not X.shape[1]:
            raise ValueError("The number of features of X_cand does not match"
                             "the number of features of X")

        if sample_weight_cand is None:
            sample_weight_cand = np.ones(len(X_cand))
        sample_weight_cand = check_array(sample_weight_cand, ensure_2d=False)
        check_consistent_length(X_cand, sample_weight_cand)

    if sample_weight is None:
        sample_weight = np.ones(y.shape)

    sample_weight = check_array(sample_weight, ensure_2d=False)
    check_consistent_length(y, sample_weight)
    if y.ndim > 1 and y.shape[1] > 1 or \
            sample_weight.ndim > 1 and sample_weight.shape[1] > 1:
        check_consistent_length(y.T, sample_weight.T)

    if X_cand is None:
        return X, y, sample_weight
    else:
        return X, y, X_cand, sample_weight, sample_weight_cand


def check_random_state(random_state, seed_multiplier=1):
    # Check given random state
    random_state = sklearn.utils.check_random_state(random_state)

    # Check multiplier
    check_scalar(seed_multiplier, 'seed_multiplier', int, min_val=1)

    seed = random_state.get_state()[1][0]

    return np.random.RandomState((seed * seed_multiplier) % (2 ** 32))
