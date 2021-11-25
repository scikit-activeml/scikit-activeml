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

    def __init__(self, x_metric, random_state=None):
        super().__init__(random_state=random_state)
        self.x_metric = x_metric

    def query(self, X_cand, X=None, batch_size=1, return_utilities=False):

        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        X: array-like (n_selected, n_features), optional (default=None)
            Selected samples. If `X` is `None`, `X` is set to an array of length
            zero.
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        return_utilities: bool, optional (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (batch_size)
            The index of the queried instance.
        utilities: np.ndarray, shape (batch_size, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """

        query_indices = np.zeros(batch_size)
        utilities = np.full((batch_size, X_cand.shape[0]), np.nan)
        n_features = X_cand.shape[1]
        n_candidates = X_cand.shape[0]
        if X is None:
            X = np.zeros((0, n_features), dtype=float)
        n_selected = X.shape[0]

        candidate_indices = np.arange(n_candidates)
        selected_indices = np.arange(n_candidates, n_candidates + n_selected)

        X_all = np.append(X_cand, X, axis=0)
        d = pairwise_distances(X_all, metric=self.x_metric)

        for i in range(batch_size):
            if selected_indices.shape[0] == 0:
                dist = d[np.ix_(candidate_indices, candidate_indices)]
                util = -np.sum(dist, axis=1)
            else:
                dist = d[np.ix_(candidate_indices, selected_indices)]
                util = np.min(dist, axis=1)
            utilities[i, candidate_indices] = util

            idx = rand_argmax(util, random_state=self.random_state)
            query_indices[i] = candidate_indices[idx]
            selected_indices = np.append(selected_indices,
                                         candidate_indices[idx], axis=0)
            candidate_indices = np.delete(candidate_indices, idx, axis=0)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
