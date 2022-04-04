import copy
import warnings
from collections.abc import Iterable

import numpy as np
from sklearn.utils.validation import (
    check_array,
    column_or_1d,
    assert_all_finite,
    check_consistent_length,
    check_random_state as check_random_state_sklearn,
)

from ._label import MISSING_LABEL, check_missing_label, is_unlabeled


def check_scalar(
        x,
        name,
        target_type,
        min_inclusive=True,
        max_inclusive=True,
        min_val=None,
        max_val=None,
):
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
        raise TypeError(
            "`{}` must be an instance of {}, not {}.".format(
                name, target_type, type(x)
            )
        )
    if min_inclusive:
        if min_val is not None and x < min_val:
            raise ValueError(
                "`{}`= {}, must be >= " "{}.".format(name, x, min_val)
            )
    else:
        if min_val is not None and x <= min_val:
            raise ValueError(
                "`{}`= {}, must be > " "{}.".format(name, x, min_val)
            )

    if max_inclusive:
        if max_val is not None and x > max_val:
            raise ValueError(
                "`{}`= {}, must be <= " "{}.".format(name, x, max_val)
            )
    else:
        if max_val is not None and x >= max_val:
            raise ValueError(
                "`{}`= {}, must be < " "{}.".format(name, x, max_val)
            )


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
        dtype = np.array(classes).dtype
        check_missing_label(missing_label, target_type=dtype, name="classes")
        n_labeled = is_unlabeled(y=classes, missing_label=missing_label).sum()
        if n_labeled > 0:
            raise ValueError(
                f"`classes={classes}` contains "
                f"`missing_label={missing_label}.`"
            )
        if cost_matrix is not None:
            check_cost_matrix(cost_matrix=cost_matrix, n_classes=len(classes))
    else:
        if cost_matrix is not None:
            raise ValueError(
                "You cannot specify 'cost_matrix' without "
                "specifying 'classes'."
            )


def check_classes(classes):
    """Check whether class labels are uniformly strings or numbers.

    Parameters
    ----------
    classes : array-like, shape (n_classes)
        Array of class labels.
    """
    if not isinstance(classes, Iterable):
        raise TypeError(
            "'classes' is not iterable. Got {}".format(type(classes))
        )
    try:
        classes_sorted = np.array(sorted(set(classes)))
        if len(classes) != len(classes_sorted):
            raise ValueError("Duplicate entries in 'classes'.")
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in classes))
        raise TypeError(
            "'classes' must be uniformly strings or numbers. Got {}".format(
                types
            )
        )


def check_class_prior(class_prior, n_classes):
    """Check if the class_prior is a valid prior.

    Parameters
    ----------
    class_prior : numeric | array_like, shape (n_classes)
        A class prior.
    n_classes : int
        The number of classes.

    Returns
    -------
    class_prior : np.ndarray, shape (n_classes)
        Numpy array as prior.
    """
    if class_prior is None:
        raise TypeError("'class_prior' must not be None.")
    check_scalar(n_classes, name="n_classes", target_type=int, min_val=1)
    if np.isscalar(class_prior):
        check_scalar(
            class_prior,
            name="class_prior",
            target_type=(int, float),
            min_val=0,
        )
        class_prior = np.array([class_prior] * n_classes)
    else:
        class_prior = check_array(class_prior, ensure_2d=False)
        is_negative = np.sum(class_prior < 0)
        if class_prior.shape != (n_classes,) or is_negative:
            raise ValueError(
                "`class_prior` must be either a non-negative"
                "float or a list of `n_classes` non-negative "
                "floats."
            )
    return class_prior.reshape(-1)


def check_cost_matrix(
        cost_matrix,
        n_classes,
        only_non_negative=False,
        contains_non_zero=False,
        diagonal_is_zero=False,
):
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
    check_scalar(n_classes, target_type=int, name="n_classes", min_val=1)
    cost_matrix_new = check_array(
        np.array(cost_matrix, dtype=float), ensure_2d=True
    )
    if cost_matrix_new.shape != (n_classes, n_classes):
        raise ValueError(
            "'cost_matrix' must have shape ({}, {}). "
            "Got {}.".format(n_classes, n_classes, cost_matrix_new.shape)
        )
    if np.sum(cost_matrix_new < 0) > 0:
        if only_non_negative:
            raise ValueError(
                "'cost_matrix' must contain only non-negative cost entries."
            )
        else:
            warnings.warn("'cost_matrix' contains negative cost entries.")
    if n_classes != 1 and np.sum(cost_matrix_new != 0) == 0:
        if contains_non_zero:
            raise ValueError(
                "'cost_matrix' must contain at least one non-zero cost "
                "entry."
            )
        else:
            warnings.warn(
                "'cost_matrix' contains contains no non-zero cost entry."
            )
    if np.sum(np.diag(cost_matrix_new) != 0) > 0:
        if diagonal_is_zero:
            raise ValueError(
                "'cost_matrix' must contain only cost entries being zero on "
                "its diagonal."
            )
        else:
            warnings.warn(
                "'cost_matrix' contains non-zero cost entries on its diagonal."
            )
    return cost_matrix_new


def check_X_y(
        X=None,
        y=None,
        X_cand=None,
        sample_weight=None,
        sample_weight_cand=None,
        accept_sparse=False,
        *,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        multi_output=False,
        allow_nan=None,
        ensure_min_samples=1,
        ensure_min_features=1,
        y_numeric=False,
        estimator=None,
        missing_label=MISSING_LABEL,
):
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

    candidates : object
        The converted and validated candidates
        Only returned if candidates is not None.

    sample_weight : np.ndarray
        The converted and validated sample_weight.

    sample_weight_cand : np.ndarray
        The converted and validated sample_weight_cand.
        Only returned if candidates is not None.
    """
    if allow_nan is None:
        allow_nan = True if missing_label is np.nan else False
    if X is not None:
        X = check_array(
            X,
            accept_sparse=accept_sparse,
            accept_large_sparse=accept_large_sparse,
            dtype=dtype,
            order=order,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=ensure_2d,
            allow_nd=allow_nd,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            estimator=estimator,
        )
    if y is not None:
        if multi_output:
            y = check_array(
                y,
                accept_sparse="csr",
                force_all_finite=True,
                ensure_2d=False,
                dtype=None,
            )
        else:
            y = column_or_1d(y, warn=True)
            assert_all_finite(y, allow_nan=allow_nan)
        if y_numeric and y.dtype.kind == "O":
            y = y.astype(np.float64)
    if X is not None and y is not None:
        check_consistent_length(X, y)
        if sample_weight is None:
            sample_weight = np.ones(y.shape)
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y, sample_weight)
        if (
                y.ndim > 1
                and y.shape[1] > 1
                or sample_weight.ndim > 1
                and sample_weight.shape[1] > 1
        ):
            check_consistent_length(y.T, sample_weight.T)

    if X_cand is not None:
        X_cand = check_array(
            X_cand,
            accept_sparse=accept_sparse,
            accept_large_sparse=accept_large_sparse,
            dtype=dtype,
            order=order,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=ensure_2d,
            allow_nd=allow_nd,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            estimator=estimator,
        )
        if X is not None and X_cand.shape[1] != X.shape[1]:
            raise ValueError(
                "The number of features of candidates does not match"
                "the number of features of X"
            )

        if sample_weight_cand is None:
            sample_weight_cand = np.ones(len(X_cand))
        sample_weight_cand = check_array(sample_weight_cand, ensure_2d=False)
        check_consistent_length(X_cand, sample_weight_cand)

    if X_cand is None:
        return X, y, sample_weight
    else:
        return X, y, X_cand, sample_weight, sample_weight_cand


def check_random_state(random_state, seed_multiplier=None):
    """Check validity of the given random state.

    Parameters
    ----------
    random_state : None | int | instance of RandomState
        If random_state is None, return the RandomState singleton used by
        np.random.
        If random_state is an int, return a new RandomState.
        If random_state is already a RandomState instance, return it.
        Otherwise raise ValueError.
    seed_multiplier : None | int, optional (default=None)
        If the random_state and seed_multiplier are not None, draw a new int
        from the random state, multiply it with the multiplier, and use the
        product as the seed of a new random state.

    Returns
    -------
    random_state: instance of RandomState
        The validated random state.
    """
    if random_state is None or seed_multiplier is None:
        return check_random_state_sklearn(random_state)

    check_scalar(
        seed_multiplier, name="seed_multiplier", target_type=int, min_val=1
    )
    random_state = copy.deepcopy(random_state)
    random_state = check_random_state_sklearn(random_state)

    seed = (random_state.randint(1, 2 ** 31) * seed_multiplier) % (2 ** 31)
    return np.random.RandomState(seed)


def check_indices(indices, A, dim="adaptive", unique=True):
    """Check if indices fit to array.

    Parameters
    ----------
    indices : array-like of shape (n_indices, n_dim) or (n_indices,)
        The considered indices, where for every `i = 0, ..., n_indices - 1`
        `indices[i]` is interpreted as an index to the array `A`.
    A : array-like
        The array that is indexed.
    dim : int or tuple of ints
        The dimensions of the array that are indexed.
        If `dim` equals `'adaptive'`, `dim` is set to first indices
        corresponding to the shape of `indices`. E.g., if `indices` is of
        shape (n_indices,), `dim` is set `0`.
    unique: bool or `check_unique`
        If `unique` is `True` unique indices are returned. If `unique` is
        `'check_unique'` an exception is raised if the indices are not unique.

    Returns
    -------
    indices: tuple of np.ndarrays or np.ndarray
        The validated indices.
    """
    indices = check_array(indices, dtype=int, ensure_2d=False)
    A = check_array(A, allow_nd=True, force_all_finite=False, ensure_2d=False)
    if unique == "check_unique":
        if indices.ndim == 1:
            n_unique_indices = len(np.unique(indices))
        else:
            n_unique_indices = len(np.unique(indices, axis=0))
        if n_unique_indices < len(indices):
            raise ValueError(
                f"`indices` contains two different indices of the "
                f"same value."
            )
    elif unique:
        if indices.ndim == 1:
            indices = np.unique(indices)
        else:
            indices = np.unique(indices, axis=0)
    check_type(dim, "dim", int, tuple, "adaptive")
    if dim == "adaptive":
        if indices.ndim == 1:
            dim = 0
        else:
            dim = tuple(range(indices.shape[1]))

    if isinstance(dim, tuple):
        for n in dim:
            check_type(n, "entry of `dim`", int)
        if A.ndim <= max(dim):
            raise ValueError(
                f"`dim` contains entry of value {max(dim)}, but all"
                f"entries of dim must be smaller than {A.ndim}."
            )
        if len(dim) != indices.shape[1]:
            raise ValueError(
                f"shape of `indices` along dimension 1 is "
                f"{indices.shape[0]}, but must be {len(dim)}"
            )
        indices = tuple(indices.T)
        for (i, n) in enumerate(indices):
            if np.any(indices[i] >= A.shape[dim[i]]):
                raise ValueError(
                    f"`indices[{i}]` contains index of value "
                    f"{np.max(indices[i])} but all indices must be"
                    f" less than {A.shape[dim[i]]}."
                )
        return indices
    else:
        if A.ndim <= dim:
            raise ValueError(
                f"`dim` has value {dim}, but must be smaller than "
                f"{A.ndim}."
            )
        if np.any(indices >= A.shape[dim]):
            raise ValueError(
                f"`indices` contains index of value "
                f"{np.max(indices)} but all indices must be"
                f" less than {A.shape[dim]}."
            )
        return indices


def check_type(obj, name, *target_types):
    """Check if obj is one of the given types.

    Parameters
    ----------
    obj: object
        The object to be checked.
    name: str
        The variable name of the object.
    target_types : iterable
        The possible types. If a target_val in `target_types` is not of type
        `type` `obj` is allowed to be equal to the target_val.

    """
    target_vals = [
        target_val
        for target_val in target_types
        if not isinstance(target_val, type)
    ]
    target_types = [
        target_type
        for target_type in target_types
        if isinstance(target_type, type)
    ]

    if (
            all(not isinstance(obj, target_type) for target_type in
                target_types)
            and obj not in target_vals
    ):

        error_str = f"`{name}` has type `{type(obj)}` but must have "
        if len(target_types) == 1:
            error_str += f"type `{target_types[0]}`"
        elif len(target_types) <= 3:
            error_str += "type "
            for i in range(len(target_types) - 1):
                error_str += f"`{target_types[i]}`,"
            error_str += f" or `{target_types[len(target_types) - 1]}`"
        else:
            error_str += f"one of the following types: {set(target_types)}"

        if len(list(target_vals)) >= 1:
            error_str += (
                f" or equal one of the following values: {set(target_vals)}"
            )
        raise TypeError(error_str + ".")


def check_bound(
        bound=None, X=None, ndim=2, epsilon=0, bound_must_be_given=False
):
    """Validates bound and returns the bound of X if bound is None.
    `bound` and `X` must not be None.

    Parameters
    ----------
    bound: array-like, shape (2, ndim), optional (default=None)
        The given bound of shape
        [[x1_min, x2_min, ..., xndim_min], [x1_max, x2_max, ..., xndim_max]]
    X: matrix-like, shape (n_samples, ndim), optional (default=None)
        The sample matrix X is the feature matrix representing samples.
    ndim: int, optional (default=2)
        The number of dimensions.
    epsilon: float, optional (default=0)
        The minimal distance between the returned bound and the values of `X`,
        if `bound` is not specified.
    bound_must_be_given: bool, optional (default=False)
        Whether it is allowed for the bound to be `None` and to be inferred by
        `X`.

    Returns
    -------
    bound: array-like, shape (2, ndim), optional (default=None)
        The given bound or bound of X.
    """

    if X is not None:
        X = check_array(X)
        if X.shape[1] != ndim:
            raise ValueError(
                f"`X` along axis 1 must be of length {ndim}. "
                f"`X` along axis 1 is of length {X.shape[1]}."
            )
    if bound is not None:
        bound = check_array(bound)
        if bound.shape != (2, ndim):
            raise ValueError(
                f"Shape of `bound` must be (2, {ndim}). "
                f"Shape of `bound` is {bound.shape}."
            )
    elif bound_must_be_given:
        raise ValueError("`bound` must not be `None`.")

    if bound is None and X is not None:
        minima = np.nanmin(X, axis=0) - epsilon
        maxima = np.nanmax(X, axis=0) + epsilon
        bound = np.append(minima.reshape(1, -1), maxima.reshape(1, -1), axis=0)
        return bound
    elif bound is not None and X is not None:
        if np.any(np.logical_or(bound[0] > X, X > bound[1])):
            warnings.warn("`X` contains values not within range of `bound`.")
        return bound
    elif bound is not None:
        return bound
    else:
        raise ValueError("`X` or `bound` must not be None.")


def check_budget_manager(
        budget,
        budget_manager,
        default_budget_manager_class,
        default_budget_manager_dict=None,
):
    """Validate if budget manager is a budgetmanager class and create a
    copy 'budget_manager_'.
    """
    if default_budget_manager_dict is None:
        default_budget_manager_dict = {}
    if budget_manager is None:
        budget_manager_ = default_budget_manager_class(
            budget=budget, **default_budget_manager_dict
        )
    else:
        if budget is not None and budget != budget_manager.budget:
            warnings.warn(
                "budgetmanager is already given such that the budget "
                "is not used. The given budget differs from the "
                "budget_managers budget."
            )
        budget_manager_ = copy.deepcopy(budget_manager)
    return budget_manager_
