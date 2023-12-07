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
    """ Typi Clust Selection

    This class implements various Typi Cluster query strategies [1], which considers
    both density and typicality of the samples.

    Parameters
    ----------
    missing_label: scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label
    random_state: int or np.random.RandomState
        The random state to use
    cluster_algo: class in sklearn.cluster (default=Kmeans)
            The cluster algorithm that to be used in the TypiClust
    k: int, optional (default=5)
            the number for knn by computation of typicality

    [1] G. Hacohen, A. Dekel, und D. Weinshall, „Active Learning on a Budget:
    Opposite Strategies Suit High and Low Budgets“, ICLR, 2022.
    """
    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        cluster_algo=KMeans,
        cluster_algo_param={},
        n_cluster_param_name="n_clusters",
        k=5
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

        self.cluster_algo = cluster_algo
        self.cluster_algo_param = cluster_algo_param
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
        """Query the next samples to be labeled

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
        y: array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label)
        candidates: None or array-like of shape (n_candidates), dtype = int or
            array-like of shape (n_candidates, n_features), optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are considered
            as candidates
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as a list of the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the candidates are
            directly given in the input candidates (not necessarily contained in X)
        batch_size: int, optional(default=1)
            The number of samples to be selects in one AL cycle.
        return_utilities: bool, optional(default=False)
            If True, also return the utilities based on the query strategy

        Returns
        ----------
        query_indices: numpy.ndarry of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected sample.
            If candidates in None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities: numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples for selecting each sample of the batch.
            Here, utilities means the typicality in the considering cluster.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate init parameter
        if not issubclass(self.cluster_algo, ClusterMixin):
            raise TypeError("Only clustering algorithm from super class sklearn.ClusterMixin is supported.")

        if not isinstance(self.k, int):
            raise TypeError("Only k as integer is supported.")

        if not isinstance(self.cluster_algo_param, dict):
            raise TypeError("Please pass a dictionary with corresponding parameter name and value in the init function.")

        if not isinstance(self.n_cluster_param_name, str):
            raise TypeError("n_cluster_param_name supports only string.")

        selected_samples = labeled_indices(y, missing_label=self.missing_label)
        n_clusters = len(selected_samples) + batch_size

        cluster_algo_param = self.cluster_algo_param.copy()
        cluster_algo_param[self.n_cluster_param_name] = n_clusters

        cluster_obj = self.cluster_algo(**cluster_algo_param)

        cluster_labels = cluster_obj.fit_predict(X)
        cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)

        covered_cluster = np.unique([cluster_labels[i] for i in selected_samples])

        cluster_sizes[covered_cluster] = 0

        if mapping is not None:
            utilities = np.zeros(shape=(batch_size, X.shape[0]))
        else:
            utilities = np.zeros(shape=(batch_size, X_cand.shape[0]))

        query_indices = []

        for i in range(batch_size):
            cluster_id = np.argmax(cluster_sizes)
            uncovered_samples_mapping = [idx for idx, value in enumerate(cluster_labels) if value == cluster_id]
            typicality = _typicality(X, uncovered_samples_mapping, self.k)
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