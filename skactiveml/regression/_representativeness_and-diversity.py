import numpy as np
from sklearn.cluster import KMeans

from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.utils import is_labeled, check_type, simple_batch
from skactiveml.utils._selection import combine_ranking


class RepresentativenessAndDiversity(SingleAnnotatorPoolQueryStrategy):
    """RD ALR, Representativeness and Diversity in active learning for
    regression

    This class implements the active learning for regression based query
    strategy RD ALR.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    qs: SingleAnnotPoolBasedQueryStrategy
        Query strategy used for further selection of the samples

    References
    ----------
    [1] Wu, Dongrui. Pool-based sequential active learning for regression. IEEE
        transactions on neural networks and learning systems, pages 1348--1359,
        2018.

    """

    def __init__(self, random_state=None, qs=None):
        super().__init__(random_state=random_state)
        self.qs = qs
        self.X_ = None
        self.k_means_ = None

    def query(
        self,
        X,
        y,
        qs_dict=None,
        fit_ensemble=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set. This data set is used for the partitioning of
            the space into clusters, by the query strategy.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        qs_dict : dict
            Dictionary for the further arguments of the query strategy besides
            `X`, `y` and `candidates`.
        fit_ensemble : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """

        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        check_type(self.qs, "self.qs", SingleAnnotatorPoolQueryStrategy)

        X_cand, mapping = self._transform_candidates(candidates, X, y)
        X_labeled = X[is_labeled(X)]
        n_labeled = len(X_labeled)

        total_query_indices = np.zeros(batch_size)
        total_utilities = np.zeros((batch_size, len(X_cand)))

        first_batch_size = np.max(0, np.min(X.shape[1] - n_labeled, batch_size))
        if n_labeled < X.shape[1]:
            X_copy = self.X_
            if X_copy is None or X_copy.shape != X.shape or np.any(X_copy != X):
                self.X_ = X.copy()
                self.k_means_ = KMeans(
                    n_clusters=n_labeled + first_batch_size,
                    random_state=self.random_state,
                )
                self.k_means_.fit(X=X, sample_weight=sample_weight)
            utilities_cand = self._k_means_utility(self.k_means_, X_cand, X_labeled)

            first_idx, first_utilities = simple_batch(
                utilities_cand, batch_size=first_batch_size, return_utilities=True
            )

            total_query_indices[:first_batch_size] = first_utilities
            total_utilities[:first_batch_size] = first_utilities

            if mapping is None:
                X_labeled = np.append(X_labeled, X_cand[first_idx], axis=0)
                X_cand = np.delete(X_cand, first_idx, axis=0)
            else:
                X_labeled = np.append(X_labeled, X[first_idx], axis=0)
                X_cand = np.delete(X_cand, mapping[first_idx], axis=0)

        second_batch_size = batch_size - first_batch_size
        if second_batch_size == 0:
            if return_utilities:
                if mapping is not None:
                    utilities = np.zeros((batch_size, len(X)))
                    utilities[mapping] = total_utilities
                    total_utilities = utilities
                return total_query_indices, total_utilities
            else:
                return total_query_indices

        if self.qs is None:
            k_means = KMeans(n_clusters=second_batch_size + n_labeled)
            k_means.fit(X=X, sample_weight=sample_weight)
            utilities_cand = self._k_means_utility(k_means, X_cand, X_labeled)
        else:
            # cluster the candidates
            n_cluster = batch_size + n_labeled
            k_means = KMeans(n_clusters=n_cluster)
            k_means.fit(X=X, sample_weight=sample_weight)
            counts = np.bincount(k_means.predict(X))
            X_cand_prediction = k_means.predict(X_cand)
            covered = k_means.predict(X_labeled)
            uncovered_clusters = ~np.isin(np.arange(n_cluster), covered)

            cluster_ranking = combine_ranking(uncovered_clusters, counts)
            sorted_clusters = np.argsort(cluster_ranking)
            n_covered = 0
            qs_utilities = np.zeros(len(X_cand))

            # decide for each cluster where to query
            for cluster in sorted_clusters:
                is_cluster_cand = X_cand_prediction == cluster
                if n_covered >= second_batch_size:
                    break
                if mapping is None:
                    cluster_candidates = X_cand[is_cluster_cand]
                else:
                    cluster_candidates = mapping[is_cluster_cand]
                _, utilities = self.qs.query(
                    X,
                    y,
                    candidates=cluster_candidates,
                    batch_size=1,
                    **qs_dict,
                )
                if mapping is None:
                    qs_utilities[is_cluster_cand] = utilities
                else:
                    idx_cluster = mapping[is_cluster_cand]
                    qs_utilities[is_cluster_cand] = utilities[idx_cluster]
                n_covered += np.sum(is_cluster_cand)

            count_utilities = counts[X_cand_prediction]
            uncovered_utilities = uncovered_clusters[X_cand_prediction]

            utilities_cand = combine_ranking(
                uncovered_utilities, count_utilities, qs_utilities
            )

        second_idx, second_utilities = simple_batch(
            utilities_cand, batch_size=first_batch_size, return_utilities=True
        )

        total_query_indices[:first_batch_size] = second_idx
        total_utilities[:first_batch_size] = second_utilities

        if return_utilities:
            return total_query_indices, total_utilities
        else:
            return total_query_indices

    def _k_means_utility(self, k_means, X_cand, X_labeled):
        clusters = k_means.predict(X_cand)
        distances = np.min(self.k_means_.transform(X_cand), axis=1)
        covered = k_means.predict(X_labeled)
        uncovered = ~np.isin(clusters, covered)
        closeness = np.exp(-distances)
        return uncovered + closeness
