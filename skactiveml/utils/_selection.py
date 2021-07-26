"""Utilities for selection."""

import numpy as np

from ._validation import check_random_state


def rand_argmin(a, random_state=None, **argmin_kwargs):
    """Returns index of minimum value. In case of ties, a randomly selected
    index of the minimum elements is returned.

    Parameters
    ----------
    a: array-like
        Indexable data-structure of whose minimum element's index is to be
        determined.
    random_state: int, RandomState instance or None, optional (default=None)
        Determines random number generation for shuffling the data. Pass an int
         for reproducible results across multiple
        function calls.
    argmin_kwargs: dict-like
        Keyword argument passed to numpy function argmin.

    Returns
    -------
    index_array: ndarray of ints
        Array of indices into the array. It has the same shape as a.shape with
        the dimension along axis removed.
    """
    random_state = check_random_state(random_state)
    a = np.asarray(a)
    index_array = np.argmax(random_state.random(a.shape) * (
            a == np.nanmin(a, **argmin_kwargs, keepdims=True)),
                            **argmin_kwargs)
    if np.isscalar(index_array) and a.ndim > 1:
        index_array = np.unravel_index(index_array, a.shape)
    index_array = np.atleast_1d(index_array)
    return index_array


def rand_argmax(a, random_state=None, **argmax_kwargs):
    """Returns index of maximum value. In case of ties, a randomly selected
    index of the maximum elements is returned.

    Parameters
    ----------
    a: array-like
        Indexable data-structure of whose maximum element's index is to be
        determined.
    random_state: int, RandomState instance or None, optional (default=None)
        Determines random number generation for shuffling the data. Pass an int
        for reproducible results across multiple function calls.
    argmax_kwargs: dict-like
        Keyword argument passed to numpy function argmin.

    Returns
    -------
    index_array: ndarray of ints
        Array of indices into the array. It has the same shape as a.shape with
        the dimension along axis removed.
    """
    random_state = check_random_state(random_state)
    a = np.asarray(a)
    index_array = np.argmax(random_state.random(a.shape) * (
            a == np.nanmax(a, **argmax_kwargs, keepdims=True)),
                            **argmax_kwargs)
    if np.isscalar(index_array) and a.ndim > 1:
        index_array = np.unravel_index(index_array, a.shape)
    index_array = np.atleast_1d(index_array)
    return index_array
