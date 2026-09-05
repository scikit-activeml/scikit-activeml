"""Shared helpers for configurable clustering algorithms."""

from copy import deepcopy
from inspect import signature


def _set_random_state_if_supported(
    cluster_algo, cluster_algo_dict, random_state
):
    """Seed a clustering algorithm unless its parameters already do so.

    Parameters
    ----------
    cluster_algo : type
        Clustering algorithm class whose constructor is inspected.
    cluster_algo_dict : dict
        Parameters that will be passed to `cluster_algo`.
    random_state : None or int or numpy.random.RandomState
        Strategy random state to propagate when supported.

    Notes
    -----
    The supplied parameter dictionary is updated in place only when the
    algorithm explicitly accepts `random_state`, the strategy has one, and
    the caller did not already provide one.
    """
    cluster_algo_params = signature(cluster_algo.__init__).parameters
    if (
        random_state is not None
        and "random_state" in cluster_algo_params
        and "random_state" not in cluster_algo_dict
    ):
        cluster_algo_dict["random_state"] = deepcopy(random_state)
