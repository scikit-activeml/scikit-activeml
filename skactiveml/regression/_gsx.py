import numpy as np
from sklearn.metrics import pairwise_distances

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.utils import rand_argmax


class GSx(SingleAnnotPoolBasedQueryStrategy):
    """Greedy Sampling on the feature space

    This class implements greedy sampling

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    """

    def __init__(self, x_metric="euclidean", random_state=None):
        super().__init__(random_state=random_state)
        self.x_metric = x_metric

    def query(self, X, y, candidates=None, batch_size=1, return_utilities=False):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
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

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        query_indices = np.zeros(batch_size)
        utilities = np.full((batch_size, X_cand.shape[0]), np.nan)

        n_features = X_cand.shape[1]
        n_candidates = X_cand.shape[0]
        if X is not None and y is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            is_labeled = np.all(~np.isnan(y), axis=1)
            X = X[is_labeled]
        elif X is None:
            X = np.zeros((0, n_features), dtype=float)
        n_selected = X.shape[0]

        candidate_indices = np.arange(n_candidates)
        selected_indices = np.arange(n_candidates, n_candidates + n_selected)

        X_all = np.append(X_cand, X, axis=0)
        d = pairwise_distances(X_all, metric=self.x_metric)

        for i in range(batch_size):
            if selected_indices.shape[0] == 0:
                dist = d[candidate_indices][candidate_indices]
                util = -np.sum(dist, axis=1)
            else:
                dist = d[candidate_indices][selected_indices]
                util = np.min(dist, axis=1)
            utilities[i, candidate_indices] = util

            idx = rand_argmax(util, random_state=self.random_state)
            query_indices[i] = candidate_indices[idx]
            selected_indices = np.append(
                selected_indices, candidate_indices[idx], axis=0
            )
            candidate_indices = np.delete(candidate_indices, idx, axis=0)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
