"""
Module implementing TypiClust Selection strategies.

TypiClust is a deep active learning strategy suited for low budgets.
Its aim is to query typical examples with the corresponding high score
of Typicality.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import MISSING_LABEL, labeled_indices
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin


class TypiClust(SingleAnnotatorPoolQueryStrategy):
    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self,
        X,
        y,
        clust_algo=KMeans,
        k=5,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate Clustering Algorithm?
        if not issubclass(clust_algo, ClusterMixin):
            raise TypeError("Only clustering algorithm from super class sklearn.ClusterMixin is supported.")

        if not isinstance(k, int):
            raise TypeError("Only k as integer is supported.")

        selected_samples = labeled_indices(y, missing_label=self.missing_label)
        n_clusters = len(selected_samples) + batch_size

        clustering_algo = clust_algo(n_clusters=n_clusters, random_state=self.random_state)

        cluster_labels = clustering_algo.fit_predict(X)
        cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)

        covered_cluster = np.unique([cluster_labels[i] for i in selected_samples])

        cluster_sizes[covered_cluster] = 0

        utilities = np.zeros(shape=(batch_size, X.shape[0]))
        query_indices = []

        for i in range(batch_size):
            cluster_id = np.argmax(cluster_sizes)
            uncovered_samples_mapping = [idx for idx, value in enumerate(cluster_labels) if value == cluster_id]
            typicality = _typicality(X, uncovered_samples_mapping, k)
            idx = np.argmax(typicality)
            typicality[selected_samples] = np.nan
            utilities[i] = typicality

            query_indices = np.append(query_indices, [idx])
            selected_samples = np.append(selected_samples, [idx])
            cluster_sizes[cluster_ids] = 0

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


def _typicality(X, uncovered_samples_mapping, k):
    typicality = np.zeros(shape=X.shape[0])
    dist_matrix = pairwise_distances(X[uncovered_samples_mapping])
    dist_matrix_sort_inc = np.sort(dist_matrix)
    knn = np.sum(dist_matrix_sort_inc[:, :k+1], axis=1)
    typi = 1 / (1 / k * knn)
    for idx, value in enumerate(uncovered_samples_mapping):
        typicality[value] = typi[idx]
    return typicality












