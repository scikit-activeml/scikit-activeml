import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans

from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.utils import is_labeled, check_type, simple_batch, MISSING_LABEL
from skactiveml.utils._selection import combine_ranking


class RepresentativenessDiversity(SingleAnnotatorPoolQueryStrategy):
    """RD ALR, Representativeness and Diversity in active learning for
    regression.

    This class implements the active learning for regression based query
    strategy RD ALR.

    Parameters
    ----------
    inner_qs: SingleAnnotPoolBasedQueryStrategy
        Query strategy used for further selection of the samples.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Wu, Dongrui. Pool-based sequential active learning for regression. IEEE
        transactions on neural networks and learning systems, pages 1348--1359,
        2018.

    """

    def __init__(
        self,
        inner_qs=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(random_state=random_state, missing_label=missing_label)
        self.inner_qs = inner_qs
        self.X_ = None
        self.k_means_ = None

    def query(
        self,
        X,
        y,
        inner_qs_dict=None,
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
            indicated by `self.missing_label`).
        inner_qs_dict : dict
            Dictionary for the further arguments of the query strategy besides
            `X`, `y` and `candidates`.
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
            contained in X).
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

        check_type(
            self.inner_qs, "self.qs", SingleAnnotatorPoolQueryStrategy, None
        )
        check_type(inner_qs_dict, "qs_dict", dict, None)
        if inner_qs_dict is None:
            inner_qs_dict = {}

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        X_labeled = X[is_labeled(y)]
        X_cand_origin = X_cand.copy()
        if mapping is not None:
            mapping_origin = mapping.copy()

        total_query_indices = np.zeros(batch_size, dtype=int)
        total_utilities = np.zeros((batch_size, len(X_cand)))
        first_batch_size = max(0, min(X.shape[1] - len(X_labeled), batch_size))
        second_batch_size = batch_size - first_batch_size

        if first_batch_size > 0:
            if self.X_ is None or not np.array_equal(X, self.X_):
                self.X_ = X.copy()
                self.k_means_ = KMeans(
                    n_clusters=min(len(X), len(X_labeled) + first_batch_size),
                    random_state=self.random_state_,
                )
                self.k_means_.fit(X=X, sample_weight=sample_weight)

            l_sample_count, _ = _cluster_sample_count(
                self.k_means_, X_labeled, X
            )
            cand_l_sample_count = l_sample_count[self.k_means_.predict(X_cand)]
            cand_closeness = _closeness_to_cluster(self.k_means_, X_cand)
            utilities_cand = combine_ranking(
                -cand_l_sample_count, cand_closeness
            )

            first_indices, first_utilities = simple_batch(
                utilities_cand,
                batch_size=first_batch_size,
                return_utilities=True,
            )

            total_query_indices[:first_batch_size] = first_indices
            total_utilities[:first_batch_size] = first_utilities

            X_labeled = np.append(X_labeled, X_cand[first_indices], axis=0)
            X_cand = np.delete(X_cand, first_indices, axis=0)

            if mapping is not None:
                mapping = np.delete(mapping, first_indices)
        else:
            first_indices = np.zeros(0, dtype=int)

        if second_batch_size > 0:
            self.k_means_ = KMeans(
                n_clusters=min(len(X), len(X_labeled) + second_batch_size),
                random_state=self.random_state_,
            )
            self.k_means_.fit(X=X, sample_weight=sample_weight)

            l_sample_count, t_sample_count = _cluster_sample_count(
                self.k_means_, X_labeled, X
            )
            cand_clusters = self.k_means_.predict(X_cand)
            cluster_ranking = rankdata(
                combine_ranking(-l_sample_count, t_sample_count), method="dense"
            )
            sorted_clusters = np.argsort(cluster_ranking)

            qs_utilities = np.zeros(len(X_cand))

            # decide for each cluster where to query
            for cluster in sorted_clusters:
                is_cluster_cand = cand_clusters == cluster

                if mapping is None:
                    cluster_candidates = X_cand[is_cluster_cand]
                else:
                    cluster_candidates = mapping[is_cluster_cand]

                if np.sum(is_cluster_cand) == 0:
                    utilities = np.zeros(0)
                elif self.inner_qs is not None:
                    utilities = self.inner_qs.query(
                        X,
                        y,
                        candidates=cluster_candidates,
                        batch_size=1,
                        return_utilities=True,
                        **inner_qs_dict,
                    )[1].flatten()
                    utilities = (
                        utilities
                        if mapping is None
                        else utilities[cluster_candidates]
                    )

                else:
                    utilities = _closeness_to_cluster(
                        self.k_means_, X_cand[is_cluster_cand]
                    )

                qs_utilities[is_cluster_cand] = utilities

            cand_cluster_ranking = cluster_ranking[cand_clusters]
            utilities_cand = cand_cluster_ranking + 1 / (
                1 + np.exp(-qs_utilities)
            )

            # batch regarding the remaining candidates after the first batch
            second_indices_off, second_utilities_off = simple_batch(
                utilities_cand,
                batch_size=second_batch_size,
                return_utilities=True,
            )
            # indices for X_cand of the remaining candidates after the first batch
            other_indices = np.delete(
                np.arange(len(X_cand_origin)), first_indices
            )
            second_indices = other_indices[second_indices_off]
            second_utilities = np.full(
                (second_batch_size, len(X_cand_origin)), np.nan
            )
            second_utilities[:, other_indices] = second_utilities_off

            total_query_indices[first_batch_size:] = second_indices
            total_utilities[first_batch_size:] = second_utilities

        if mapping is None:
            utilities = total_utilities
            query_indices = total_query_indices
        else:
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping_origin] = total_utilities
            query_indices = mapping_origin[total_query_indices]

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


def _cluster_sample_count(k_means, X_labeled, X):
    labeled_samples_per_cluster = np.bincount(
        k_means.predict(X_labeled)
        if len(X_labeled) != 0
        else np.zeros(0, dtype=int),
        minlength=k_means.n_clusters,
    )
    total_samples_per_cluster = np.bincount(
        k_means.predict(X), minlength=k_means.n_clusters
    )
    return labeled_samples_per_cluster, total_samples_per_cluster


def _closeness_to_cluster(k_means, X_cand):
    cand_distances = np.min(k_means.transform(X_cand), axis=1)
    cand_closeness = -cand_distances
    return cand_closeness
