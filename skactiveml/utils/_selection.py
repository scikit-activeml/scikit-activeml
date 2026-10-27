"""Utilities for selection."""

import warnings

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
    index_array : numpy.ndarray of ints
        Indices of the selected elements, always at least one-dimensional.
        For multi-dimensional `a`, `axis=None` and `keepdims=False` return
        coordinates of one element, with shape `(a.ndim,)`. Otherwise, the
        shape follows `numpy.argmin`, including `keepdims`. With `axis=None`
        and `keepdims=True`, the index refers to the flattened array.
    """
    random_state = check_random_state(random_state)
    a = np.asarray(a)
    axis = argmin_kwargs.get("axis")
    index_array = np.argmax(
        random_state.random(a.shape)
        * (a == np.nanmin(a, axis=axis, keepdims=True)),
        **argmin_kwargs,
    )
    if np.ndim(index_array) == 0 and a.ndim > 1:
        index_array = np.unravel_index(index_array, a.shape)
    return np.atleast_1d(index_array)


def rand_argmax(a, random_state=None, **argmax_kwargs):
    """Returns index of maximum value. In case of ties, a randomly selected
    index of the maximum elements is returned.

    Parameters
    ----------
    a : array-like
        Indexable data-structure of whose maximum element's index is to be
        determined.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling the data. Pass an int
        for reproducible results across multiple function calls.
    argmax_kwargs : dict-like
        Keyword argument passed to numpy function `argmax`.

    Returns
    -------
    index_array : numpy.ndarray of ints
        Indices of the selected elements, always at least one-dimensional.
        For multi-dimensional `a`, `axis=None` and `keepdims=False` return
        coordinates of one element, with shape `(a.ndim,)`. Otherwise, the
        shape follows `numpy.argmax`, including `keepdims`. With `axis=None`
        and `keepdims=True`, the index refers to the flattened array.
    """
    random_state = check_random_state(random_state)
    a = np.asarray(a)
    axis = argmax_kwargs.get("axis")
    index_array = np.argmax(
        random_state.random(a.shape)
        * (a == np.nanmax(a, axis=axis, keepdims=True)),
        **argmax_kwargs,
    )
    if np.ndim(index_array) == 0 and a.ndim > 1:
        index_array = np.unravel_index(index_array, a.shape)
    return np.atleast_1d(index_array)


def simple_batch(
    utilities,
    random_state=None,
    batch_size=1,
    return_utilities=False,
    method="max",
):
    """Select a batch of entries from the given utilities without replacement.

    Each entry represents one action. For example, a utility matrix can
    represent sample-annotator pairs. Selection is global over all entries,
    and only selected entries are excluded from subsequent batch steps.

    Parameters
    ----------
    utilities : np.ndarray
        The utilities of the actions. `np.nan` excludes an entry. For
        proportional selection, utilities must be nonnegative, with at least
        `batch_size` positive entries after reducing the batch size to the
        number of available entries. If no entries are available, the result
        is empty.
    random_state : int, RandomState instance or None, default=None
        The random state to use.
    batch_size : int, default=1
        The number of entries to select. If larger than the number of
        available entries, it is reduced with a warning.
    return_utilities : bool, default=False
        If True, the utilities are returned.
    method : str, default='max'
        'max' selects the largest utilities. 'proportional' samples entries
        with probabilities proportional to their utilities.

    Returns
    -------
    best_indices : np.ndarray of shape (batch_size,) if utilities.ndim == 1 \
            else (batch_size, utilities.ndim)
        The indices of the selected entries, using one coordinate per utility
        dimension for multidimensional input.
    batch_utilities : np.ndarray of shape (batch_size, *utilities.shape)
        Utilities before each batch selection, with previously selected
        entries set to `np.nan` (if `return_utilities=True`).

    """
    # validation
    utilities = check_array(
        utilities,
        ensure_2d=False,
        dtype=float,
        ensure_all_finite="allow-nan",
        allow_nd=True,
        copy=True,
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
        p = np.nan_to_num(utilities.ravel(), nan=0.0)
        if np.any(p < 0):
            raise ValueError(
                "`utilities` must be nonnegative for proportional selection."
            )
        if np.count_nonzero(p) < batch_size:
            raise ValueError(
                "Proportional selection requires at least `batch_size` "
                "positive utilities."
            )
        if batch_size > 0:
            # Scale before summing to avoid overflow for finite weights.
            p /= p.max()
            p /= p.sum()
            flat_indices = random_state.choice(
                p.size, size=batch_size, p=p, replace=False
            )
            best_indices = np.column_stack(
                np.unravel_index(flat_indices, utilities.shape)
            )

        batch_utilities = np.repeat([utilities], batch_size, axis=0)
        for i in range(batch_size):
            batch_utilities[(i,) + tuple(best_indices[:i].T)] = np.nan
    else:
        raise ValueError(
            f'"method" has to be either "max" or "proportional" '
            f"but {method} was given."
        )

    if utilities.ndim == 1:
        best_indices = best_indices.flatten()

    if return_utilities:
        return best_indices, batch_utilities
    else:
        return best_indices


def combine_ranking(*iter_ranking, rank_method=None, rank_per_batch=False):
    """Combine rankings in order of priority.

    Earlier rankings take precedence over later rankings. Later rankings
    break ties in earlier rankings.

    Parameters
    ----------
    iter_ranking : iterable of array-like
        One or more numerical rankings with the same number of dimensions
        and shapes that can be broadcast together. An entry containing
        `np.nan` in any ranking is excluded and receives `np.nan` in the
        result. Infinite values are valid ranking values.
    rank_method : {'average', 'min', 'max', 'dense', 'ordinal'}, default=None
        How to assign ranks to entries tied in every input ranking, following
        the `method` argument of `scipy.stats.rankdata`. `None` uses 'dense'.
        With 'ordinal', ties follow the flattened input order.
    rank_per_batch : bool, default=False
        If `True`, the first axis identifies independent batches and the
        remaining axes are flattened within each batch. Otherwise, all
        entries are ranked together.

    Returns
    -------
    combined_ranking : np.ndarray, dtype=float64
        The combined ranks in the broadcast shape, with larger values
        indicating higher priority. A single input ranking is converted to
        float and returned without reranking.
    """

    if rank_method is None:
        rank_method = "dense"
    check_type(rank_method, "rank_method", str)
    if rank_method not in ("average", "min", "max", "dense", "ordinal"):
        raise ValueError(f"Unknown rank method '{rank_method}'.")
    check_type(rank_per_batch, "rank_per_batch", bool)

    rankings = [
        check_array(
            ranking, allow_nd=True, ensure_2d=False, ensure_all_finite=False
        )
        for ranking in iter_ranking
    ]
    if not rankings:
        raise ValueError("At least one ranking is required.")
    if any(ranking.ndim != rankings[0].ndim for ranking in rankings[1:]):
        raise ValueError(
            "All rankings must have the same number of dimensions."
        )
    if len(rankings) == 1:
        return rankings[0].astype(float)

    rankings = np.broadcast_arrays(*rankings)
    shape = rankings[0].shape
    n_batches = shape[0] if rank_per_batch else 1
    rankings = [ranking.reshape(n_batches, -1) for ranking in rankings]
    is_valid = ~np.logical_or.reduce(
        [np.isnan(ranking) for ranking in rankings]
    )
    combined_ranking = np.full(rankings[0].shape, np.nan)

    # Putting invalid entries last keeps them out of valid tie counts.
    order = np.lexsort([*reversed(rankings), ~is_valid], axis=1)
    new_group = np.ones(order.shape, dtype=bool)
    new_group[:, 1:] = False
    for ranking in rankings:
        values = np.take_along_axis(ranking, order, axis=1)
        new_group[:, 1:] |= values[:, 1:] != values[:, :-1]
    ranks = np.cumsum(new_group, axis=1)
    if rank_method != "dense":
        ranks = rankdata(ranks, method=rank_method, axis=1)
    np.put_along_axis(combined_ranking, order, ranks, axis=1)
    combined_ranking[~is_valid] = np.nan

    return combined_ranking.reshape(shape)
