import numpy as np

from sklearn.utils import check_array, check_scalar
from ..base import SingleAnnotPoolBasedQueryStrategy


class RandomSampler(SingleAnnotPoolBasedQueryStrategy):
    """Random sampling.

    This class implements random sampling

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    """

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    def query(self, X_cand, batch_size=1, return_utilities=False, **kwargs):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples
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
        # Check the given data
        X_cand = check_array(X_cand, force_all_finite=False)

        # Check 'batch_size'
        check_scalar(batch_size, 'batch_size', int, min_val=1)

        utilities = self.random_state.random_sample(len(X_cand))

        best_indices = utilities.argsort()[-batch_size:][::-1]

        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices
