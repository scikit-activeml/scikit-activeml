"""Utilities for selection."""

import operator
import warnings
from functools import reduce

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_array

from ._validation import check_random_state, check_scalar, check_type


def rand_argmin(a, random_state=None, **argmin_kwargs):
    """Returns index of minimum value. In case of ties, a randomly selected
    index of the minimum elements is returned.

    Parameters
    ----------
    a : array-like
        Indexable data-structure of whose minimum element's index is to be
        determined.
    random_state : int or RandomState instance or None, default=None
        Determines random number generation for shuffling the data. Pass an int
        for reproducible results across multiple function calls.
    argmin_kwargs : dict-like
        Keyword argument passed to numpy function `argmin`.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along axis removed.
    """
    random_state = check_random_state(random_state)
    a = np.asarray(a)
    index_array = np.argmax(
        random_state.random(a.shape)
        * (a == np.nanmin(a, **argmin_kwargs, keepdims=True)),
        **argmin_kwargs,
    )
    if np.isscalar(index_array) and a.ndim > 1:
        index_array = np.unravel_index(index_array, a.shape)
    index_array = np.atleast_1d(index_array)
    return index_array


def rand_argmax(a, random_state=None, **argmax_kwargs):
    """Returns index of maximum value. In case of ties, a randomly selected
    index of the maximum elements is returned.

    Parameters
    ----------
     : array-like
        Indexable data-structure of whose maximum element's index is to be
        determined.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling the data. Pass an int
        for reproducible results across multiple function calls.
    argmax_kwargs : dict-like
        Keyword argument passed to numpy function `argmax`.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along axis removed.
    """
    random_state = check_random_state(random_state)
    a = np.asarray(a)
    index_array = np.argmax(
        random_state.random(a.shape)
        * (a == np.nanmax(a, **argmax_kwargs, keepdims=True)),
        **argmax_kwargs,
    )
    if np.isscalar(index_array) and a.ndim > 1:
        index_array = np.unravel_index(index_array, a.shape)
    index_array = np.atleast_1d(index_array)
    return index_array


def simple_batch(
    utilities,
    random_state=None,
    batch_size=1,
    return_utilities=False,
    method="max",
):
    """Generates a batch by selecting the highest values in the `utilities`.
    If `utilities` is an ND-array, the returned utilities will be an
    (N+1)D-array, with the shape `batch_size` x `len(utilities)`, filled the
    given `utilities` but set the n-th highest values in the n-th row to
    `np.nan`.

    Parameters
    ----------
    utilities : np.ndarray
        The utilities to be used to create the batch.
    random_state : int, RandomState instance or None, default=None
        The random state to use.
    batch_size : int, default=1
        The number of samples to be selected in one AL cycle.
    return_utilities : bool, default=False
        If True, the utilities are returned.
    method : str, default='max'
        Determines how to select 'best_indices'. 'max' selects the indices with
        the maximum utilities. 'proportional' randomly choose the
        'best_indices' with the probabilities proportional to 'utilities'.

    Returns
    -------
    best_indices : np.ndarray of shape (batch_size,) if utilities.ndim == 1 \
            else (batch_size, utilities.ndim)
        The indices of the batch samples.
    batch_utilities : np.ndarray of shape (batch_size, len(utilities))
        The `utilities` of the batch (if `return_utilities=True`).

    """
    # validation
    utilities = check_array(
        utilities,
        ensure_2d=False,
        dtype=float,
        ensure_all_finite="allow-nan",
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

    check_type(method, "method", str)

    # generate batch
    best_indices = np.empty((batch_size, utilities.ndim), dtype=int)
    if method == "max":
        batch_utilities = np.empty((batch_size,) + utilities.shape)
        for i in range(batch_size):
            best_indices[i] = rand_argmax(utilities, random_state=random_state)
            batch_utilities[i] = utilities
            utilities[tuple(best_indices[i])] = np.nan
    elif method == "proportional":
        random_state = check_random_state(random_state)
        p = utilities / np.nansum(utilities)
        p[np.isnan(p)] = 0
        best_indices = random_state.choice(
            len(utilities),
            size=batch_size,
            p=p,
            replace=False,
        )

        batch_utilities = np.repeat([utilities], batch_size, axis=0)
        for i in range(batch_size):
            batch_utilities[i, best_indices[:i]] = np.nan
    else:
        raise ValueError(
            f'"method" has to be either "max" or "proportional" '
            f"but {method} was given."
        )

    if utilities.ndim == 1:
        best_indices = best_indices.flatten()

    # Check whether utilities are to be returned.
    if return_utilities:
        return best_indices, batch_utilities
    else:
        return best_indices


def combine_ranking(*iter_ranking, rank_method=None, rank_per_batch=False):
    """Combine different rankings hierarchically to one ranking assignment.
    A ranking index `i` is ranked higher than index `j` iff
    `ranking[i] > ranking[j]`. For the combined ranking it will hold that the
    first ranking of `iter_ranking` always determines the ranking position at
    an index, and only when two ranking assignments are equal the second
    ranking will determine the ranking position and so forth.

    Parameters
    ----------
    iter_ranking : iterable of array-like
        The different rankings. They must share a common shape in the sense
        that they have the same number of dimensions and are broadcastable by
        numpy.
    rank_method : string, default=None
        The method by which the utilities are ranked. See `scipy.rankdata`s
        argument `method` for details.
    rank_per_batch : bool, default=False
        Whether the first index determines the batch and is not used for
        ranking.

    Returns
    -------
    combined_ranking : np.ndarray
        The combined ranking.
    """

    if rank_method is None:
        rank_method = "dense"
    check_type(rank_method, "rank_method", str)
    check_type(rank_per_batch, "rank_per_batch", bool)

    iter_ranking = list(iter_ranking)
    for idx, ranking in enumerate(iter_ranking):
        iter_ranking[idx] = check_array(
            ranking, allow_nd=True, ensure_2d=False, ensure_all_finite=False
        ).astype(float)
        if idx != 0 and iter_ranking[idx - 1].ndim != ranking.ndim:
            raise ValueError(
                f"The number of dimensions of the `ranking` in "
                f"`iter_ranking` must be the same, but "
                f"`iter_ranking[{idx}].ndim == {ranking.ndim}"
                f" and `iter_ranking[{idx-1}].ndim == "
                f"{iter_ranking[idx - 1].ndim}`."
            )
    np.broadcast_shapes(*(u.shape for u in iter_ranking))

    combined_ranking = iter_ranking[0]

    for idx in range(1, len(iter_ranking)):
        next_ranking = iter_ranking[idx]
        cr_shape = combined_ranking.shape
        if rank_per_batch:
            rank_shape = (
                cr_shape[0],
                max(reduce(operator.mul, cr_shape[1:], 1), 1),
            )
            rank_dict = {"method": rank_method, "axis": 1}
        else:
            rank_shape = reduce(operator.mul, cr_shape, 1)
            rank_dict = {"method": rank_method}

        combined_ranking = combined_ranking.reshape(rank_shape)

        # exchange nan values to make rankdata work.
        nan_values = np.isnan(combined_ranking)
        combined_ranking[nan_values] = -np.inf
        combined_ranking = rankdata(combined_ranking, **rank_dict).astype(
            float
        )
        combined_ranking[nan_values] = np.nan
        combined_ranking = combined_ranking.reshape(cr_shape)

        combined_ranking = combined_ranking + 1 / (
            1 + np.exp(-next_ranking)
        )  # sigmoid

    return combined_ranking
