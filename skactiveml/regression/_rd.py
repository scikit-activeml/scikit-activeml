import numpy as np
from sklearn.cluster import KMeans

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.utils import is_labeled, check_type, simple_batch


class RD(SingleAnnotPoolBasedQueryStrategy):
    """RD

    This class implements the active learning for regression based query
    strategy RD.

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

    def query(self, X, y, qs_dict=None, fit_ensemble=True, sample_weight=None,
              candidates=None, batch_size=1, return_utilities=False):
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

        check_type(self.qs, 'self.qs', SingleAnnotPoolBasedQueryStrategy)

        X_cand, mapping = self._transform_candidates(candidates, X, y)
        X_labeled = X[is_labeled(X)]
        n_labeled = len(X_labeled)

        if mapping is None:
            total_utilities = np.zeros((batch_size, len(X_cand)))
        else:
            total_utilities = np.zeros((batch_size, len(X)))

        first_batch_size = np.max(0, np.min(X.shape[1] - n_labeled, batch_size))
        if n_labeled < X.shape[1]:
            X_copy = self.X_
            if X_copy is None or X_copy.shape != X.shape or np.any(X_copy != X):
                self.X_ = X.copy()
                self.k_means_ = KMeans(n_clusters=n_labeled + first_batch_size,
                                       random_state=self.random_state)
                self.k_means_.fit(X=X, sample_weight=sample_weight)
            utilities_cand = self._k_means_utility(self.k_means_, X_cand,
                                                   X_labeled)
            if mapping is None:
                utilities = utilities_cand
            else:
                utilities = np.full(len(X), np.nan)
                utilities[mapping] = utilities_cand

            first_idx, first_utilities = simple_batch(
                utilities, batch_size=first_batch_size, return_utilities=True)

            total_utilities[:first_batch_size] = first_utilities

            if mapping is None:
                X_labeled = np.append(X_labeled, X_cand[first_idx], axis=0)
                X_cand = np.delete(X_cand, first_idx, axis=0)
            else:
                X_labeled = np.append(X_labeled, X[first_idx], axis=0)
                X_cand = np.delete(X_cand, mapping[first_idx], axis=0)

        second_batch_size = batch_size - first_batch_size
        if self.qs is None:
            k_means = KMeans(n_clusters=second_batch_size + n_labeled)
            k_means.fit(X=X, sample_weight=sample_weight)
            utilities_cand = self._k_means_utility(k_means, X_cand, X_labeled)
        else:
            k_means = KMeans(n_clusters=batch_size + n_labeled)
            k_means.fit(X=X, sample_weight=sample_weight)




        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(utilities, return_utilities=return_utilities)

    def _k_means_utility(self, k_means, X_cand, X_labeled):
        clusters = k_means.predict(X_cand)
        distances = np.min(self.k_means_.transform(X_cand), axis=1)
        covered = k_means.predict(X_labeled)
        uncovered = ~np.isin(clusters, covered)
        closeness = np.exp(-distances)
        return uncovered + closeness
