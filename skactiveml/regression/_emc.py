from copy import deepcopy

import numpy as np

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.utils import rand_argmax


class EMC(SingleAnnotPoolBasedQueryStrategy):
    """Expected Model Change

    This class implements greedy sampling

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_bootstraps: int, optional (default=1)
        The minimum number of samples the estimator requires.
    ord: int or string (default=2)
        The Norm to measure the gradient.
    """

    def __init__(self, k_bootstraps=10, ord=2, random_state=None):
        super().__init__(random_state=random_state)
        self.ord = ord
        self.k_bootstraps = k_bootstraps

    def query(self, X_cand, reg, X, y, batch_size=1, return_utilities=False):

        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        X: array-like, shape (n_samples, n_features)
            Complete training data set.
        reg: SkactivemlRegressor
            regressor to predict values of X_cand.
        y: array-like, shape (n_samples)
            Values of the training data set.
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

        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        k = min(n_samples, self.k_bootstraps)
        learners = [deepcopy(reg) for _ in range(k)]
        sample_indices = np.arange(n_samples)
        subsets_indices = [rng.choice(sample_indices, size=n_samples*(k-1)//k)
                           for _ in range(k)]

        for learner, subset_indices in zip(learners, subsets_indices):
            X_for_learner = X[subset_indices]
            y_for_learner = y[subset_indices]
            learner.fit(X_for_learner, y_for_learner)

        results = np.array([learner.predict(X_cand) for learner in learners])
        scalars = np.average(np.abs(results), axis=0)
        norms = np.linalg.norm(X_cand, ord=self.ord, axis=1)
        utilities = np.multiply(scalars, norms)

        if return_utilities:
            return rand_argmax(utilities), utilities
        else:
            return rand_argmax(utilities)
