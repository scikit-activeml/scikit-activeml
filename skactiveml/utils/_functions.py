import warnings
import numpy as np
from sklearn.utils import check_array

from skactiveml.utils import check_scalar#, rand_argmax


def initialize_class_with_kwargs(class_obj, **kwargs):
    parameters = class_obj.__init__.__code__.co_varnames
    kwargs = dict(filter(lambda e: e[0] in parameters, kwargs.items()))
    return class_obj(**kwargs)


def simple_batch(
        utilities, random_state, batch_size=1, return_utilities=False):
    """Generates a batch by selecting the highest values in the 'utilities'.
    The returned utilities will be an 2D-array with the shape batch_size x
    len(utilities), filled the given utilities but set the n-th highest values
    in the n-th row to np.nan.

    Parameters
    ----------
    utilities : np.ndarray
        The utilities to be used to create the batch.
    random_state : numeric | np.random.RandomState
        The random state to use.
    batch_size : int, optional (default=1)
        The number of samples to be selected in one AL cycle.
    return_utilities : bool (default=False)
        If True, the utilities are returned.

    Returns
    -------
    best_indices : np.ndarray, shape (batch_size)
        The index of the batch instance.
    batch_utilities : np.ndarray,  shape (batch_size, len(utilities))
        The utilities of the batch (if return_utilities=True).

    """
    # validation
    utilities = check_array(utilities, ensure_2d=False, dtype=float)
    if batch_size == 'adaptive':
        batch_size = 1
    check_scalar(batch_size, target_type=int, name='batch_size',
                 min_val=1)
    if len(utilities) < batch_size:
        warnings.warn(
            "'batch_size={}' is larger than number of candidate samples "
            "in 'utilities'. Instead, 'batch_size={}' was set.".format(
                batch_size, len(utilities)))
        batch_size = len(utilities)
    # generate batch
    batch_utilities = np.empty((batch_size, len(utilities)))
    best_indices = np.empty(batch_size, dtype=int)
    for i in range(batch_size):
        # TODO change np.argmax to rand_argmax
        best_indices[i] = np.nanargmax(
            [utilities], axis=1)#, random_state=random_state)
        batch_utilities[i] = utilities
        utilities[best_indices[i]] = np.nan
    # Check whether utilities are to be returned.
    if return_utilities:
        return best_indices, batch_utilities
    else:
        return best_indices
