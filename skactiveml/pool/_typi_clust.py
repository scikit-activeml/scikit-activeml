"""
Module implementing TypiClust.

TypiClust is a deep active learning strategy suited for low budgets.
Its aim is to query typical examples with the corresponding high score
of 'typicality'.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import MISSING_LABEL, labeled_indices, check_scalar, rand_argmax
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class TypiClust(SingleAnnotatorPoolQueryStrategy):
    """Typical Clustering (TypiClust)

    This class implements the Typical Clustering (TypiClust) query strategy
    [1]_, which considers both diversity and typicality (representativeness) of
    the samples.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.
    cluster_algo : ClusterMixin.__class__, default=KMeans
        The cluster algorithm to be used.
    cluster_algo_dict : dict, optional (default=None)
        The parameters passed to the clustering algorithm `cluster_algo`,
        excluding the parameter for the number of clusters.
    n_cluster_param_name : string, default="n_clusters"
        The name of the parameter for the number of clusters.
    k : int, default=5
        The number for k-nearest-neighbors for the computation of typicality.

    References
    ----------
    .. [1] G. Hacohen, A. Dekel, and D. Weinshall. Active Learning on a Budget:
       Opposite Strategies Suit High and Low Budgets. In Int. Conf. Mach.
       Learn., pages 8175–8195, 2022.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        cluster_algo=KMeans,
        cluster_algo_dict=None,
        n_cluster_param_name="n_clusters",
        k=5,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.cluster_algo = cluster_algo
        self.cluster_algo_dict = cluster_algo_dict
        self.n_cluster_param_name = n_cluster_param_name
        self.k = k

    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
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
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        _, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=True
        )

        # Validate init parameter
        check_scalar(self.k, "k", target_type=int, min_val=1)

        if not (
            isinstance(self.cluster_algo_dict, dict)
            or self.cluster_algo_dict is None
        ):
            raise TypeError(
                "Please pass a dictionary with corresponding parameter name "
                "and value in the `init` function."
            )
        cluster_algo_dict = (
            {}
            if self.cluster_algo_dict is None
            else self.cluster_algo_dict.copy()
        )

        if not isinstance(self.n_cluster_param_name, str):
            raise TypeError("`n_cluster_param_name` supports only string.")

        labeled_sample_indices = labeled_indices(
            y, missing_label=self.missing_label
        )

        n_clusters = len(labeled_sample_indices) + batch_size
        cluster_algo_dict[self.n_cluster_param_name] = n_clusters
        cluster_obj = self.cluster_algo(**cluster_algo_dict)

        cluster_labels = cluster_obj.fit_predict(X)

        # determine number of samples per cluster and mask clusters with
        # labeled samples
        cluster_sizes = np.zeros(n_clusters)
        cluster_ids, cluster_ids_sizes = np.unique(
            cluster_labels, return_counts=True
        )
        cluster_sizes[cluster_ids] = cluster_ids_sizes
        covered_cluster = np.unique(
            [cluster_labels[i] for i in labeled_sample_indices]
        )
        if len(covered_cluster) > 0:
            cluster_sizes[covered_cluster] = 0

        utilities = np.full(shape=(batch_size, X.shape[0]), fill_value=np.nan)
        query_indices = []
        for i in range(batch_size):
            if cluster_sizes.max() == 0:
                typicality = np.ones(len(X))
            else:
                cluster_id = rand_argmax(
                    cluster_sizes, random_state=self.random_state_
                )
                is_cluster = cluster_labels == cluster_id
                uncovered_samples_mapping = np.where(is_cluster)[0]
                typicality = _typicality(X, uncovered_samples_mapping, self.k)
            utilities[i, mapping] = typicality[mapping]
            utilities[i, query_indices] = np.nan
            idx = rand_argmax(
                typicality[mapping], random_state=self.random_state_
            )
            idx = mapping[idx[0]]

            query_indices = np.append(query_indices, [idx]).astype(int)
            cluster_sizes[cluster_id] = 0

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


def _typicality(X, uncovered_samples_mapping, k, eps=1e-7):
    """
    Calculation the typicality of samples `X` in uncovered clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    uncovered_samples_mapping : np.ndarray of shape (n_candidates,),
    default=None
       Index array that maps `candidates` to `X_for_cluster`.
    k : int
        k for computation of k nearst neighbors.
    eps : float > 0, default=1e-7
        Minimum distance sum to compute typicality.

    Returns
    -------
    typicality : numpy.ndarray of shape (n_X)
        The typicality of all uncovered samples in X
    """
    typicality = np.full(shape=X.shape[0], fill_value=-np.inf)
    if len(uncovered_samples_mapping) == 1:
        typicality[uncovered_samples_mapping] = 1
        return typicality
    k = np.min((len(uncovered_samples_mapping) - 1, k))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X[uncovered_samples_mapping])
    dist_matrix_sort_inc, _ = nn.kneighbors(
        X[uncovered_samples_mapping], n_neighbors=k + 1, return_distance=True
    )
    knn = np.sum(dist_matrix_sort_inc, axis=1) + eps
    typi = ((1 / k) * knn) ** (-1)
    typicality[uncovered_samples_mapping] = typi
    return typicality
