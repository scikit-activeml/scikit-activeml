import inspect
import operator
import warnings
from functools import reduce

import numpy as np
from scipy.stats import rankdata

from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    column_or_1d,
    _is_arraylike,
)

from ._selection import rand_argmax
from ._validation import check_scalar, check_indices, check_type


def call_func(f_callable, only_mandatory=False, **kwargs):
    """Calls a function with the given parameters given in kwargs if they
    exist as parameters in f_callable.

    Parameters
    ----------
    f_callable : callable
        The function or object that is to be called
    only_mandatory : boolean
        If True only mandatory parameters are set.
    kwargs : kwargs
        All parameters that could be used for calling f_callable.

    Returns
    -------
    called object
    """
    params = inspect.signature(f_callable).parameters
    param_keys = params.keys()
    if only_mandatory:
        param_keys = list(
            filter(lambda k: params[k].default == inspect._empty, param_keys)
        )

    vars = dict(filter(lambda e: e[0] in param_keys, kwargs.items()))

    return f_callable(**vars)


def simple_batch(utilities, random_state=None, batch_size=1, return_utilities=False):
    """Generates a batch by selecting the highest values in the 'utilities'.
    If utilities is an ND-array, the returned utilities will be an (N+1)D-array,
    with the shape batch_size x utilities.shape, filled the given utilities but
    set the n-th highest values in the n-th row to np.nan.

    Parameters
    ----------
    utilities : np.ndarray
        The utilities to be used to create the batch.
    random_state : numeric | np.random.RandomState (default=None)
        The random state to use. If `random_state is None` random `random_state`
        is used.
    batch_size : int, optional (default=1)
        The number of samples to be selected in one AL cycle.
    return_utilities : bool (default=False)
        If True, the utilities are returned.

    Returns
    -------
    best_indices : np.ndarray, shape (batch_size) if ndim == 1
        (batch_size, ndim) else
        The index of the batch instance.
    batch_utilities : np.ndarray,  shape (batch_size, len(utilities))
        The utilities of the batch (if return_utilities=True).

    """
    # validation
    utilities = check_array(
        utilities,
        ensure_2d=False,
        dtype=float,
        force_all_finite="allow-nan",
        allow_nd=True,
    )
    check_scalar(batch_size, target_type=int, name="batch_size", min_val=1)
    max_batch_size = np.sum(~np.isnan(utilities), dtype=int)
    if max_batch_size < batch_size:
        warnings.warn(
            "'batch_size={}' is larger than number of candidate samples "
            "in 'utilities'. Instead, 'batch_size={}' was set.".format(
                batch_size, max_batch_size
            )
        )
        batch_size = max_batch_size
    # generate batch

    batch_utilities = np.empty((batch_size,) + utilities.shape)
    best_indices = np.empty((batch_size, utilities.ndim), dtype=int)

    for i in range(batch_size):
        best_indices[i] = rand_argmax(utilities, random_state=random_state)
        batch_utilities[i] = utilities
        utilities[tuple(best_indices[i])] = np.nan
    # Check whether utilities are to be returned.
    if utilities.ndim == 1:
        best_indices = best_indices.flatten()

    if return_utilities:
        return best_indices, batch_utilities
    else:
        return best_indices


def update_X_y(X, y, y_update, idx_update=None, X_update=None):
    """Update the training data by the updating samples/labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set.
    y : array-like of shape (n_samples)
        Labels of the training data set.
    idx_update : array-like of shape (n_updates) or int
        Index of the samples or sample to be updated.
    X_update : array-like of shape (n_updates, n_features) or (n_features)
        Samples to be updated or sample to be updated.
    y_update : array-like of shape (n_updates) or numeric
        Updating labels or updating label.

    Returns
    -------
    X_new : np.ndarray of shape (n_new_samples, n_features)
        The new training data set.
    y_new : np.ndarray of shape (n_new_samples)
        The new labels.
    """
    X = check_array(X)
    y = column_or_1d(check_array(y, force_all_finite=False, ensure_2d=False))
    check_consistent_length(X, y)

    if _is_arraylike(y_update):
        y_update = check_array(y_update, force_all_finite=False, ensure_2d=False)
        y_update = column_or_1d(y_update)
    else:
        check_scalar(y_update, "y_update", (int, float))
        y_update = np.array([y_update])

    if idx_update is not None:
        if not _is_arraylike(idx_update):
            check_scalar(idx_update, "idx_update", int)
            idx_update = np.array([idx_update])
        idx_update = check_indices(idx_update, A=X, unique="check_unique")
        check_consistent_length(y_update, idx_update)
        X_new = X.copy()
        y_new = y.copy()
        y_new[idx_update] = y_update
        return X_new, y_new
    elif X_update is not None:
        X_update = check_array(X_update, ensure_2d=False)
        if X_update.ndim == 1:
            X_update = X_update.reshape(1, -1)
        check_consistent_length(X.T, X_update.T)
        check_consistent_length(y_update, X_update)
        X_new = np.append(X, X_update, axis=0)
        y_new = np.append(y, y_update, axis=0)
        return X_new, y_new
    else:
        raise ValueError("`idx_update` or `X_update` must not be `None`")


def rank_utilities(*iter_utilities, rank_method=None, rank_per_batch=False):
    """Combine different utilities hierarchically to one utility assignment.

    Parameters
    ----------
    iter_utilities : iterable of array-like
        The different utilities. They must share a common shape in the sense
        that the have the same number of dimensions and are broadcastable by
        numpy. If the common shape has more than two dimensions.
    rank_method : string, optional (default = None)
        The method by which the utilities are ranked. See `scipy.rankdata`s
        argument `method` for details.
    rank_per_batch: bool, optional (default = False)
        Whether the last index is the batch size and is not used for ranking

    Returns
    -------
    combined_utilities : np.ndarray
        The combined utilities.
    """

    if rank_method is None:
        rank_method = "min"
    check_type(rank_method, "rank_method", str)
    check_type(rank_per_batch, "rank_per_batch", bool)

    iter_utilities = list(iter_utilities)
    for idx, utilities in enumerate(iter_utilities):
        iter_utilities[idx] = check_array(
            utilities, allow_nd=True, ensure_2d=False, force_all_finite=False
        ).astype(float)
        if idx != 0 and iter_utilities[idx - 1].ndim != utilities.ndim:
            raise ValueError(
                f"The number of dimensions of the `utilities` in "
                f"`iter_utilities` must be the same, but "
                f"`iter_utilities[{idx}].ndim == {utilities.ndim}"
                f" and `iter_utilities[{idx-1}].ndim == "
                f"{iter_utilities[idx-1].ndim}`."
            )
    np.broadcast_shapes(*(u.shape for u in iter_utilities))

    iter_utilities = [i for i in reversed(iter_utilities)]
    combined_utilities = iter_utilities[0]

    for idx in range(1, len(iter_utilities)):
        next_utilities = iter_utilities[idx]
        nu_shape = next_utilities.shape
        if rank_per_batch:
            rank_shape = (nu_shape[0], max(reduce(operator.mul, nu_shape[1:], 1), 1))
            rank_dict = {"method": rank_method, "axis": 1}
        else:
            rank_shape = reduce(operator.mul, nu_shape, 1)
            rank_dict = {"method": rank_method}

        rank_next_utilities = next_utilities.reshape(rank_shape)
        # exchange nan values to make `rank_data` work.
        nan_values = np.isnan(rank_next_utilities)
        rank_next_utilities[nan_values] = -np.inf
        rank_next_utilities = rankdata(rank_next_utilities, **rank_dict).astype(float)
        rank_next_utilities[nan_values] = np.nan
        next_utilities = rank_next_utilities.reshape(nu_shape)

        if combined_utilities.min() != combined_utilities.max():
            combined_utilities = (
                1
                / (combined_utilities.max() - combined_utilities.min() + 1)
                * (combined_utilities - combined_utilities.min())
            )

        combined_utilities = combined_utilities + next_utilities

    return combined_utilities
