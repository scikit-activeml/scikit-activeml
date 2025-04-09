"""
Module implementing `ProbCover`, which is a deep active learning strategy
suited for low budgets.
"""

import numpy as np
import warnings

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.utils.validation import column_or_1d

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    MISSING_LABEL,
    rand_argmax,
    check_scalar,
)


class ProbCover(SingleAnnotatorPoolQueryStrategy):
    """Probability Coverage (ProbCover)

    This class implements the Probability Coverage (ProbCover) query strategy
    [1]_, which aims at maximizing the probability coverage in a meaningful
    sample embedding space.

    Parameters
    ----------
    n_classes : None or int, default=None
        This parameter is used to determine the delta value. If
        `n_classes=None`, the number of classes is extracted from the
        given labels. If this extracted number of classes is below 2,
        `n_classes=2` is used as a fallback.
    deltas : None or array-like of shape (n_deltas,), default=None
        List of deltas (ball radii) to be tested for finding the maximum
        value satisfying a sample coverage >= `alpha`. If no value in
        `deltas` satisfies this constraint, a warning is raised where
        the minimum `delta` value is used. If `deltas=None`, the values
        `np.arange(0.1, 2.1, 0.1)` are used.
    alpha : float in (0, 1), alpha=0.95
        Minimum coverage as a constraint for the `delta` selection.
    cluster_algo : ClusterMixin.__class__, default=sklearn.cluster.KMeans
        The cluster algorithm to be used for determining the best delta value.
    cluster_algo_dict : dict, default=None
        The parameters passed to the clustering algorithm `cluster_algo`,
        excluding the parameter for the number of clusters.
    n_cluster_param_name : string, default="n_clusters"
        The name of the parameter for the number of clusters.
    distance_func : callable, default=sklearn.metrics.pairwise_distances
        Takes as input `X` to compute the distances between each pair of
        samples. This function can also only return the precomputed distances
        of each pair in `X` for speedup.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] O. Yehuda, A. Dekel, G. Hacohen, and D. Weinshall. Active Learning
       Through a Covering Lens. In Adv. Neural Inf. Process. Syst., 2022.
    """

    def __init__(
        self,
        n_classes=None,
        deltas=None,
        alpha=0.95,
        cluster_algo=KMeans,
        cluster_algo_dict=None,
        n_cluster_param_name="n_clusters",
        distance_func=pairwise_distances,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.deltas = deltas
        self.alpha = alpha
        self.n_classes = n_classes
        self.cluster_algo = cluster_algo
        self.cluster_algo_dict = cluster_algo_dict
        self.n_cluster_param_name = n_cluster_param_name
        self.distance_func = distance_func

    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        update=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. The indexing refers to the samples in `X`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan. The indexing
            refers to the samples in `X`.
        """
        # Check parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        _, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=True
        )
        is_candidate = np.full(len(X), fill_value=False)
        is_candidate[mapping] = True
        n_classes = self.n_classes
        if n_classes is None:
            n_classes = max(len(np.unique(y[~is_candidate])), 2)
        check_scalar(
            n_classes,
            "n_classes",
            min_val=2,
            min_inclusive=True,
            target_type=int,
        )
        if self.deltas is None:
            deltas = np.arange(0.2, 2.2, 0.2)
        else:
            deltas = column_or_1d(self.deltas, dtype=float)
            deltas = np.sort(deltas)
            if (deltas < 0).any():
                raise ValueError("`deltas` must contain non-negative floats.")
        check_scalar(
            self.alpha,
            "alpha",
            min_val=0,
            max_val=1,
            min_inclusive=False,
            max_inclusive=False,
            target_type=float,
        )
        if not (
            isinstance(self.cluster_algo_dict, dict)
            or self.cluster_algo_dict is None
        ):
            raise TypeError(
                "Pass a dictionary with corresponding parameter names and "
                "values according to the `init` function of `cluster_algo`."
            )
        cluster_algo_dict = (
            {}
            if self.cluster_algo_dict is None
            else self.cluster_algo_dict.copy()
        )
        check_scalar(update, name="update", target_type=bool)

        if update or not hasattr(self, "delta_max_"):
            # Compute distances between each pair of observed samples.
            self.distances_ = self.distance_func(X)

            # Compute the maximum `delta` value satisfying a purity >= `alpha`.
            self.delta_max_ = deltas[0]
            max_purity = -1
            if len(deltas) > 1:
                cluster_algo_dict[self.n_cluster_param_name] = n_classes
                cluster_obj = self.cluster_algo(**cluster_algo_dict)
                y_cluster = cluster_obj.fit_predict(X)
                is_impure = y_cluster[:, None] != y_cluster
                for delta in deltas:
                    edges = self.distances_ <= delta
                    purity = 1 - (edges * is_impure).any(axis=1).mean()
                    max_purity = max(max_purity, purity)
                    if purity < self.alpha:
                        break
                    self.delta_max_ = delta

            # Check whether condition defined by `alpha` was satisfied.
            if max_purity < self.alpha:
                warnings.warn(
                    f"The maximum purity was {max_purity} being smaller "
                    f"than the required value `alpha={self.alpha}`. You must"
                    f"provide smaller values in `deltas` to avoid "
                    f"this warning."
                )

        # Compute edges of the graph with the samples as vertices.
        edges = self.distances_ <= self.delta_max_

        # Perform sample-wise selection of the batch.
        query_indices = np.full(batch_size, fill_value=-1, dtype=int)
        utilities = np.full((batch_size, len(X)), fill_value=np.nan)
        for b in range(batch_size):
            # Step (ii) in [1]: Remove incoming edges for covered samples.
            is_covered = edges[~is_candidate].any(axis=0)
            edges[:, is_covered] = False
            # Step (i) in [1]: Query the sample with the highest out-degree.
            utilities[b][is_candidate] = edges[is_candidate].sum(axis=1)
            idx = rand_argmax(utilities[b], random_state=self.random_state_)[0]
            is_candidate[idx] = False
            query_indices[b] = idx

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
