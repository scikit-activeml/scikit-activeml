from copy import deepcopy

import numpy as np
from sklearn.utils.validation import _is_arraylike, check_random_state

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy, SkactivemlRegressor
from skactiveml.utils import simple_batch


class QBC(SingleAnnotPoolBasedQueryStrategy):
    """Regression based on Query by Committee

    This class implements an Regression adaption of Query by Committee. It
    tries to estimate the model variance by a Committee of estimators.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_bootstraps: int, optional (default=1)
        The number of members in a committee.

    References
    ----------
    [1] Burbidge, Robert and Rowland, Jem J and King, Ross D. Active learning
        for regression based on query by committee. International conference on
        intelligent data engineering and automated learning, pages 209--218,
        2007.

    """

    def __init__(self, random_state=None, k_bootstraps=5):
        super().__init__(random_state=random_state)
        self.k_bootstraps = k_bootstraps

    def query(self, X_cand, ensemble, X, y, batch_size=1,
              return_utilities=False):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        X: array-like, shape (n_samples, n_features)
            Complete training data set.
        ensemble: {SkactivemlRegressor, array-like}
            Regressor to predict the data.
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

        if isinstance(ensemble, SkactivemlRegressor) and \
                hasattr(ensemble, 'n_estimators'):
            if hasattr(ensemble, 'estimators_'):
                est_arr = ensemble.estimators_
            else:
                est_arr = [ensemble] * ensemble.n_estimators
        elif _is_arraylike(ensemble):
            est_arr = deepcopy(ensemble)
        else:
            raise TypeError(
                f'`ensemble` must either be a `{SkactivemlRegressor} '
                f'with the attribute `n_esembles` and `estimators_` after '
                f'fitting or a list of {SkactivemlRegressor} objects.'
            )

        random_state = check_random_state(self.random_state)
        n_samples = len(X)
        k = min(n_samples, self.k_bootstraps)
        sample_indices = np.arange(n_samples)
        random_state.shuffle(sample_indices)
        subsets_indices = np.array_split(sample_indices, k)

        for learner, subset_indices in zip(est_arr, subsets_indices):
            X_for_learner = X[subset_indices]
            y_for_learner = y[subset_indices]
            learner.fit(X_for_learner, y_for_learner)

        results = np.array([learner.predict(X_cand) for learner in est_arr])
        utilities = np.std(results, axis=0)

        return simple_batch(utilities, return_utilities=return_utilities)

    # TODO: https://aclanthology.org/D08-1112.pdf




