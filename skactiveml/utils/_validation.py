import copy
import warnings
import numbers
import numpy as np

from inspect import Parameter, signature
from sklearn.utils.validation import (
    check_array,
    column_or_1d,
    assert_all_finite,
    check_consistent_length,
    check_random_state as check_random_state_sklearn,
    _check_n_features as sklearn_check_n_features,
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
    min_inclusive : bool, default=True
        If `True`, the minimum valid value is inclusive, otherwise exclusive.
    max_inclusive : bool, default=True
        If `True`, the maximum valid value is inclusive, otherwise exclusive.
    min_val : float or int, default=None
        The minimum valid value the parameter can take. If `None` (default), it
        is implied that the parameter does not have a lower bound.
    max_val : float or int, default=None
        The maximum valid value the parameter can take. If `None` (default), it
        is implied that the parameter does not have an upper bound.

    Raises
    ------
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
        if min_val is not None and (x < min_val or np.isnan(x)):
            raise ValueError(
                "`{}`= {}, must be >= " "{}.".format(name, x, min_val)
            )
    else:
        if min_val is not None and (x <= min_val or np.isnan(x)):
            raise ValueError(
                "`{}`= {}, must be > " "{}.".format(name, x, min_val)
            )

    if max_inclusive:
        if max_val is not None and (x > max_val or np.isnan(x)):
            raise ValueError(
                "`{}`= {}, must be <= " "{}.".format(name, x, max_val)
            )
    else:
        if max_val is not None and (x >= max_val or np.isnan(x)):
            raise ValueError(
                "`{}`= {}, must be < " "{}.".format(name, x, max_val)
            )


def _is_nonstring_iterable(obj):
    """Check whether `obj` is iterable but not string-like.

    Parameters
    ----------
    obj : object
        Object to be checked.

    Returns
    -------
    is_iterable : bool
        `True` if `obj` is iterable and not a string, bytes object, or
        NumPy string scalar, and `False` otherwise.
    """
    if isinstance(obj, (str, bytes, np.str_)):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _has_nested_classes(classes):
    """Check whether `classes` contains one vocabulary per target label.

    Parameters
    ----------
    classes : object
        Candidate class specification. Single-output class specifications are
        expected to be one-dimensional iterables of scalar labels. Nested
        specifications are expected to contain one iterable per target label.

    Returns
    -------
    has_nested_classes : bool
        `True` if `classes` has a nested structure and `False` otherwise.

    Raises
    ------
    ValueError
        If `classes` is an empty iterable.
    """
    if classes is None:
        return False
    if not _is_nonstring_iterable(classes):
        return False

    # Materialize once (handles generators).
    outer = list(classes)
    if len(outer) == 0:
        raise ValueError("`classes` must not be empty.")

    nested = [_is_nonstring_iterable(value) for value in outer]
    if any(nested) and not all(nested):
        raise ValueError(
            "`classes` must be uniformly flat or nested; mixed class "
            "vocabularies are not supported."
        )
    return all(nested)


def _check_1d_class_list(c, name="classes"):
    """Validate a one-dimensional class list.

    Parameters
    ----------
    c : iterable
        Class labels of a single output.
    name : str, default="classes"
        Name used in error messages.

    Raises
    ------
    TypeError
        If `c` is not iterable, contains unhashable or unsupported label
        types, or mixes numeric and string labels.
    ValueError
        If `c` is empty, not one-dimensional, or contains duplicate labels.
    """
    if not _is_nonstring_iterable(c):
        raise TypeError(f"`{name}` must be iterable. Got {type(c)}.")

    arr = np.asarray(list(c), dtype=object)

    if arr.ndim != 1:
        raise ValueError(
            f"`{name}` must be one-dimensional. Got shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty.")

    # Ensure scalars are hashable and unique.
    try:
        if len(set(arr.tolist())) != arr.size:
            raise ValueError(f"Duplicate entries in `{name}`.")
    except TypeError as e:
        raise TypeError(
            f"`{name}` must contain hashable scalar labels "
            f"(strings or numbers)."
        ) from e

    # Enforce "uniformly strings or numbers"
    kinds = set()
    for v in arr.tolist():
        if isinstance(v, (str, np.str_)):
            kinds.add("str")
        elif isinstance(v, (numbers.Number, np.number)):
            kinds.add("num")
        else:
            raise TypeError(
                f"`{name}` must contain only strings or numbers. "
                f"Got element {v!r} of type {type(v)}."
            )

    if len(kinds) != 1:
        raise TypeError(
            f"`{name}` must be uniformly strings or numbers. "
            f"Got mixture: {sorted(kinds)}."
        )


def _canonicalize_multilabel_probas(
    probas,
    n_samples=None,
    n_outputs=None,
    allow_none=False,
):
    """Convert multilabel probabilities to a 2D positive-class matrix.

    Parameters
    ----------
    probas : array-like of shape (n_samples, n_outputs) or list of \
            array-like of shape (n_samples, 2), or None
        Multilabel probabilities. A two-dimensional array-like is interpreted
        as one positive-class probability per label. A list whose entries are
        two-dimensional is interpreted as one binary probability matrix per
        label.
    n_samples : int or None, default=None
        Expected number of samples. If not `None`, the returned array must
        have this many rows.
    n_outputs : int or None, default=None
        Expected number of outputs. If not `None`, the returned array must
        have this many columns.
    allow_none : bool, default=False
        If `True`, `None` is returned unchanged.

    Returns
    -------
    probas : numpy.ndarray of shape (n_samples, n_outputs) or None
        Canonicalized multilabel probabilities containing one positive-class
        probability per output.

    Raises
    ------
    ValueError
        If `probas` is `None` while `allow_none=False`, or if the provided
        probabilities do not match the expected multilabel format.
    """
    if probas is None:
        if allow_none:
            return None
        raise ValueError("`probas` must not be `None`.")

    is_per_output_list = isinstance(probas, list) and any(
        np.asarray(probas_j).ndim == 2 for probas_j in probas
    )
    if is_per_output_list:
        if n_outputs is not None and len(probas) != n_outputs:
            raise ValueError(
                f"`probas` contains {len(probas)} outputs, expected "
                f"{n_outputs}."
            )

        probas_cols = []
        for j, probas_j in enumerate(probas):
            probas_j = np.asarray(probas_j, dtype=float)
            if probas_j.ndim != 2 or probas_j.shape[1] != 2:
                raise ValueError(
                    f"`probas[{j}]` must have shape `(n_samples, 2)`, got "
                    f"{probas_j.shape}."
                )
            if n_samples is not None and probas_j.shape[0] != n_samples:
                raise ValueError(
                    f"`probas[{j}]` has {probas_j.shape[0]} samples, "
                    f"expected {n_samples}."
                )
            probas_cols.append(probas_j[:, 1])

        probas = np.column_stack(probas_cols)
    else:
        probas = np.asarray(probas, dtype=float)
        if probas.ndim != 2:
            raise ValueError(
                "`probas` must have shape `(n_samples, n_outputs)` for "
                f"multilabel data, got {probas.shape}."
            )

    if n_samples is not None and probas.shape[0] != n_samples:
        raise ValueError(
            f"`probas` has {probas.shape[0]} samples, expected {n_samples}."
        )
    if n_outputs is not None and probas.shape[1] != n_outputs:
        raise ValueError(
            f"`probas` has {probas.shape[1]} outputs, expected {n_outputs}."
        )

    return probas


def check_classes(classes):
    """Check whether class labels are uniformly strings or numbers.

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of such array-likes, \
            default=None
        The classes labels (single output setting), or a list of arrays of
        class labels (multioutput setting).
    """
    if classes is None:
        return

    if not _is_nonstring_iterable(classes):
        raise TypeError(f"`classes` is not iterable. Got {type(classes)}.")

    if _has_nested_classes(classes):
        outer = list(classes)
        for i, c in enumerate(outer):
            _check_1d_class_list(c, name=f"classes[{i}]")
    else:
        _check_1d_class_list(classes, name="classes")


def check_classifier_params(classes, missing_label, cost_matrix=None):
    """Check whether the general classifier are compatible with each other.

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of such array-likes, \
            default=None
        The classes labels (single output setting), or a list of arrays of
        class labels (multioutput setting).
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label. In the case of a multioutput
        setting, we expect that the missing label is identical across all
        tasks.
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost matrix to quantify costs of misclassifications.

        - Checked only for single output.
        - Must be `None` for multioutput.
    """
    check_missing_label(missing_label)

    if classes is None:
        if cost_matrix is not None:
            raise ValueError(
                "You cannot specify `cost_matrix` without specifying "
                "`classes`."
            )
        return

    # Validates structure + duplicates + type-uniformity.
    check_classes(classes)

    # Check whether `classes` contains one vocabulary per target label.
    has_nested_classes = _has_nested_classes(classes)

    # Enforce cost_matrix semantics.
    if has_nested_classes:
        if cost_matrix is not None:
            raise ValueError(
                "`cost_matrix` must be `None` when `classes` contains "
                "per-output vocabularies."
            )
        outer = list(classes)
        # Missing_label type check and ensure missing_label not in any task's
        # classes.
        for i, c in enumerate(outer):
            c_arr = np.asarray(list(c))
            check_missing_label(
                missing_label, target_type=c_arr.dtype, name=f"classes[{i}]"
            )
            n_unlabeled = is_unlabeled(
                y=c_arr, missing_label=missing_label
            ).sum()
            if n_unlabeled > 0:
                raise ValueError(
                    f"`classes[{i}]={list(c)}` contains "
                    f"`missing_label={missing_label}`."
                )
    else:
        c_arr = np.asarray(list(classes))
        check_missing_label(
            missing_label, target_type=c_arr.dtype, name="classes"
        )
        n_unlabeled = is_unlabeled(y=c_arr, missing_label=missing_label).sum()
        if n_unlabeled > 0:
            raise ValueError(
                f"`classes={list(classes)}` contains "
                f"`missing_label={missing_label}`."
            )
        if cost_matrix is not None:
            check_cost_matrix(cost_matrix=cost_matrix, n_classes=len(c_arr))


def check_class_prior(class_prior, n_classes):
    """Check if the `class_prior` is a valid prior.

    Parameters
    ----------
    class_prior : numeric or array_like of shape (n_classes,)
        A class prior.
    n_classes : int
        The number of classes.

    Returns
    -------
    class_prior : np.ndarray of shape (n_classes,)
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
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix.
    n_classes : int
        Number of classes.
    only_non_negative : bool, default=False
        This parameter determines whether the matrix must contain only
        non-negative cost entries.
    contains_non_zero : bool, default=False
        This parameter determines whether the matrix must contain at least on
        non-zero cost entry.
    diagonal_is_zero : bool, default=False
        This parameter determines whether the diagonal cost entries must be
        zero.

    Returns
    -------
    cost_matrix_new : np.ndarray of shape (n_classes, n_classes)
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
    ensure_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    target_type="single-output",
    allow_nan=None,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
    missing_label=MISSING_LABEL,
):
    """Input validation for standard estimators. Adjusted from `sklearn` [1]_.

    Checks X and y for consistent length, enforces `X` to be at least 2D and
    `y` 1D. By default, `X` is checked to be non-empty and containing only
    finite values. Standard input checks are also applied to `y`, such as
    checking that `y` does not have `np.nan` or `np.inf` targets.
    For multi-label `y`, set `target_type="multi-label"`.
    If the dtype of `X` is object, attempt converting to float, raising on
    failure.

    Parameters
    ----------
    X : nd-array or list or sparse matrix, default=None
        Labeled input data.
    y : nd-array or list or sparse matrix, default=None
        Labels for X.
    X_cand : nd-array or list or sparse matrix, default=None
        Unlabeled input data
    sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
    sample_weight_cand : array-like of shape (n_candidates,), default=None
            Sample weights of the candidates.
    accept_sparse : string or boolean or list of string, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : 'F', or 'C' or None, default=None
        Whether an array will be forced to be fortran or c-style.
    copy : boolean, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    ensure_all_finite : boolean or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.
    ensure_2d : boolean, default=True
        Whether to raise a value error if X is not 2D.
    allow_nd : boolean, default=False
        Whether to allow X.ndim > 2.
    target_type : "single-output" or "multi-label" or "multi-output", \
            default="single-output"
        Resolved target type controlling target-array validation.
    allow_nan : boolean, default=None
        Whether to allow np.nan in y.
    ensure_min_samples : int, default=1
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and `ensure_2d` is True. Setting to 0 disables
        this check.
    y_numeric : boolean, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
    missing_label : scalar or string or np.nan or None, default=np.nan
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

    References
    ----------
    .. [1] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
       Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg,
       J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E.
       Duchesnay. Scikit-learn: Machine Learning in Python. J. Mach. Learn.
       Res., 12:2825–2830, 2011.
    """
    if allow_nan is None:
        allow_nan = (
            True
            if isinstance(missing_label, float) and np.isnan(missing_label)
            else False
        )
    if X is not None:
        X = check_array(
            X,
            accept_sparse=accept_sparse,
            accept_large_sparse=accept_large_sparse,
            dtype=dtype,
            order=order,
            copy=copy,
            ensure_all_finite=ensure_all_finite,
            ensure_2d=ensure_2d,
            allow_nd=allow_nd,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            estimator=estimator,
        )
    if y is not None:
        if target_type not in {
            "single-output",
            "multi-label",
            "multi-output",
        }:
            raise ValueError(
                "`target_type` must be one of {'single-output', "
                "'multi-label', 'multi-output'}."
            )
        if target_type in {"multi-label", "multi-output"}:
            y = check_array(
                y,
                accept_sparse="csr",
                ensure_all_finite=True,
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
            ensure_all_finite=ensure_all_finite,
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
    random_state : None or int or instance of RandomState
        - If `random_state` is None, return the `RandomState` singleton used by
          `np.random`.
        - If `random_state` is an int, return a new `RandomState`.
        - If random_state is already a `RandomState` instance, return it.
        - Otherwise raise `ValueError`.
    seed_multiplier : None or int, default=None
        If the `random_state` and `seed_multiplier` are not `None`, draw a new
        int from the random state, multiply it with the multiplier, and use the
        product as the seed of a new random state.

    Returns
    -------
    random_state : instance of RandomState
        The validated random state.
    """
    if random_state is None or seed_multiplier is None:
        return check_random_state_sklearn(random_state)

    check_scalar(
        seed_multiplier, name="seed_multiplier", target_type=int, min_val=1
    )
    random_state = copy.deepcopy(random_state)
    random_state = check_random_state_sklearn(random_state)

    seed = (random_state.randint(1, 2**31) * seed_multiplier) % (2**31)
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
    dim : int or tuple of ints or 'adaptive', default='adaptive'
        The dimensions of the array that are indexed.
        If `dim` equals `'adaptive'`, `dim` is set to first indices
        corresponding to the shape of `indices`. E.g., if `indices` is of
        shape (n_indices,), `dim` is set `0`.
    unique : bool or 'check_unique', default=True
        If `unique` is `True` unique indices are returned. If `unique` is
        `'check_unique'` an exception is raised if the indices are not unique.

    Returns
    -------
    indices : tuple of np.ndarray or np.ndarray
        The validated indices.
    """
    indices = check_array(indices, dtype=int, ensure_2d=False)
    A = check_array(
        A, allow_nd=True, ensure_all_finite=False, ensure_2d=False, dtype=None
    )
    if unique == "check_unique":
        if indices.ndim == 1:
            n_unique_indices = len(np.unique(indices))
        else:
            n_unique_indices = len(np.unique(indices, axis=0))
        if n_unique_indices < len(indices):
            raise ValueError(
                "`indices` contains two different indices of the "
                "same value."
            )
    elif unique:
        if indices.ndim == 1:
            indices = np.unique(indices)
        else:
            indices = np.unique(indices, axis=0)
    check_type(dim, "dim", int, tuple, target_vals=["adaptive"])
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
        for i, n in enumerate(indices):
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


def check_type(
    obj, name, *target_types, target_vals=None, indicator_funcs=None
):
    """Check whether an object satisfies type, value, or indicator rules.

    Parameters
    ----------
    obj : object
        The object to be checked.
    name : str
        The variable name of the object.
    target_types : iterable
        The possible types.
    target_vals : iterable, default=None
        Possible further values that the object is allowed to equal.
    indicator_funcs : iterable, default=None
        Possible further custom indicator (boolean) functions that accept
        the object by returning `True` if the object is passed as a parameter.

    Raises
    ------
    TypeError
        If `obj` does not match any allowed type or value and is not accepted
        by any indicator function.
    """
    target_vals = target_vals if target_vals is not None else []
    indicator_funcs = indicator_funcs if indicator_funcs is not None else []

    wrong_type = not isinstance(obj, target_types)
    wrong_value = obj not in target_vals
    wrong_index = all(not i_func(obj) for i_func in indicator_funcs)

    if wrong_type and wrong_value and wrong_index:
        error_str = f"`{name}` "
        if len(target_types) == 0 and len(target_vals) == 0:
            error_str += " must"
        if len(target_vals) == 0 and len(target_types) > 0:
            error_str += f" has type `{type(obj)}`, but must"
        elif len(target_vals) > 0 and len(target_types) == 0:
            error_str += f" has value `{obj}`, but must"
        else:
            error_str += f" has type `{type(obj)}` and value `{obj}`, but must"

        if len(target_types) == 1:
            error_str += f" have type `{target_types[0]}`"
        elif 1 <= len(target_types) <= 3:
            error_str += " have type"
            for i in range(len(target_types) - 1):
                error_str += f" `{target_types[i]}`,"
            error_str += f" or `{target_types[len(target_types) - 1]}`"
        elif len(target_types) > 3:
            error_str += (
                f" have one of the following types: {set(target_types)}"
            )

        if len(target_vals) > 0:
            if len(target_types) > 0 and len(indicator_funcs) == 0:
                error_str += " or"
            elif len(target_types) > 0 and len(indicator_funcs) > 0:
                error_str += ","
            error_str += (
                f" equal one of the following values: {set(target_vals)}"
            )

        if len(indicator_funcs) > 0:
            if len(target_types) > 0 or len(target_vals) > 0:
                error_str += " or"
            error_str += (
                f" be accepted by one of the following custom boolean "
                f"functions: {set(i_f.__name__ for i_f in indicator_funcs)}"
            )

        raise TypeError(error_str + ".")


def _check_callable(func, name, n_positional_parameters=None):
    """Check whether `func` is callable with the expected arity.

    Parameters
    ----------
    func : callable
        Callable to be validated.
    name : str
        Name used in error messages.
    n_positional_parameters : int, default=None
        Expected number of positional parameters without defaults. If `None`,
        one positional parameter is expected.

    Raises
    ------
    TypeError
        If `func` is not callable.
    ValueError
        If `func` does not expose the expected number of positional
        parameters.
    """
    if n_positional_parameters is None:
        n_positional_parameters = 1

    if not callable(func):
        raise TypeError(
            f"`{name}` must be callable. " f"`{name}` is of type {type(func)}"
        )

    # count the number of arguments that have no default value
    n_actual_positional_parameters = len(
        list(
            filter(
                lambda x: x.default == Parameter.empty,
                signature(func).parameters.values(),
            )
        )
    )

    if n_actual_positional_parameters != n_positional_parameters:
        raise ValueError(
            f"The number of positional parameters of the callable has to "
            f"equal {n_positional_parameters}. "
            f"The number of positional parameters is "
            f"{n_actual_positional_parameters}."
        )


def check_bound(
    bound=None, X=None, ndim=2, epsilon=0, bound_must_be_given=False
):
    """Validates `bound` and returns the `bound` of `X` if `bound` is `None`.
    `bound` or `X` must not be None.

    Parameters
    ----------
    bound: array-like of shape (2, ndim), default=None
        The given bound of shape
        [[x1_min, x2_min, ..., xndim_min], [x1_max, x2_max, ..., xndim_max]]
    X: matrix-like of shape (n_samples, ndim), default=None
        `X` is the feature matrix representing samples.
    ndim: int, default=2
        The number of dimensions.
    epsilon: float, default=0
        The minimal distance between the returned bound and the values of `X`,
        if `bound` is not specified.
    bound_must_be_given: bool, default=False
        Whether it is allowed for the `bound` to be `None` and to be inferred
        by `X`.

    Returns
    -------
    bound : array-like of shape (2, ndim), default=None
        The given `bound` or bound of `X`.
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
    """Validate if `budget_manager` is a budget manager class and create a
    copy `budget_manager_`.

    Parameters
    ----------
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be queried, with
        0 <= budget <= 1. See Also :class:`skactiveml.base.BudgetManager`.
    budget_manager : BudgetManager, default=None
        Budget manager to be checked. If `budget_manager` is `None`, a new
        budget manager using the class `default_budget_manager_class` is
        created using the `default_budget_manager_dict` as parameters.
    default_budget_manager_class : BudgetManager.__class__
        Fallback class for creation of a budget manger (cf. description of
        `budget_manager`).
    default_budget_manager_dict : dict, default=None
        Fallback parameters for the creation of a budget manger (cf.
        description of `budget_manager`).

    Returns
    -------
    budget_manager_ : BudgetManager
        Checked or newly created budget manager object.
    """
    uses_rand = (
        "random_state" in signature(default_budget_manager_class).parameters
    )
    if default_budget_manager_dict is None:
        default_budget_manager_dict = {}
    elif not uses_rand:
        default_budget_manager_dict.pop("random_state", None)
    if budget_manager is None:
        budget_manager_ = default_budget_manager_class(
            budget=budget,
            **default_budget_manager_dict,
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


def check_n_features(obj, X, reset):
    """Validate and update the expected number of features of an estimator.

    Parameters
    ----------
    obj : object
        Estimator-like object expected to expose `n_features_in_`.
    X : array-like of shape (n_samples, n_features)
        Input data whose second dimension determines the number of features.
    reset : bool
        If `True`, set `obj.n_features_in_` from `X`. If `False`, validate
        `X` against the existing value of `obj.n_features_in_`.

    Raises
    ------
    ValueError
        If `reset=False` and `X` does not match the stored number of features.
    """
    if reset:
        obj.n_features_in_ = X.shape[1] if len(X) > 0 else None
    elif not reset:
        if obj.n_features_in_ is not None:
            sklearn_check_n_features(obj, X, reset=reset)


def _check_forward_outputs(forward_outputs):
    """Validate the `forward_outputs` mapping used by `SkorchMixin`.

    Parameters
    ----------
    forward_outputs : dict[str, tuple[int, Callable | None]]
        Mapping that describes how to obtain and post-process the outputs of
        `module.forward` for prediction.

        Given `raw_outputs = module.forward(X)`, each entry
        `name -> (idx, transform)` is interpreted as:

        - `idx`: integer index of `raw_outputs` (0-based).
        - `transform`: callable `f(tensor) -> tensor` or `None`.
          If `transform` is not `None`, it is applied to the selected
          raw tensor; otherwise the raw tensor is used.

    Raises
    ------
    TypeError
        If `forward_outputs` is not a dictionary or contains invalid entry
        specifications.
    ValueError
        If `forward_outputs` is empty or contains negative indices.
    """

    # Check forward_outputs configured.
    check_type(forward_outputs, "forward_outputs", dict)
    if len(forward_outputs) == 0:
        raise ValueError("`forward_outputs` must contain at least one entry.")

    # Validate and normalize specs into a dict:
    # name -> (idx, transform)
    for name, spec in forward_outputs.items():
        if (
            not isinstance(spec, tuple)
            or len(spec) != 2
            or not isinstance(spec[0], int)
        ):
            raise TypeError(
                "Each value in forward_outputs must be a tuple "
                "`(idx: int, transform: Callable | None)`. "
                f"Got {spec!r} for key {name!r}."
            )
        idx, transform = spec
        if idx < 0:
            raise ValueError(
                f"Index `idx={idx}` for key {name!r} must be " f"non-negative."
            )
        if transform is not None and not callable(transform):
            raise TypeError(
                "The second element of each forward_outputs tuple "
                f"must be a `Callable` or `None`. Got {transform!r} "
                f"for key {name!r}."
            )
