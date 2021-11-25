import numpy as np

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.pool.regression._gsx import GSx


class GSy(SingleAnnotPoolBasedQueryStrategy):
    """Greedy Sampling on the feature space

    This class implements greedy sampling

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_0: int, optional (default=1)
        The minimum number of samples the estimator requires.
    """

    def __init__(self, x_metric=None, y_metric=None, k_0=1, random_state=None):
        super().__init__(random_state=random_state)
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.k_0 = k_0

    def query(self, X_cand, reg=None, X=None, y=None, batch_size=1,
              return_utilities=False):

        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        X: array-like, shape (n_samples, n_features), optional (default=None)
            Complete training data set.
        reg: SkactivemlEstimator, optional (default=None)
            estimator to estimate values of X_cand.
        y: array-like, shape (n_samples), optional (default=None)
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
        n_train_samples = X.shape[0]
        reg.fit(X, y)
        y_pred = reg.predict(X_cand)

        n_samples = X_cand.shape[0]

        batch_size_x = max(0, min(self.k_0 - n_train_samples, batch_size))
        batch_size_y = batch_size - batch_size_x

        query_indices = np.zeros((batch_size,))
        utilities = np.zeros((batch_size, n_samples))

        if batch_size_x > 0:
            gs = GSx(x_metric=self.x_metric, random_state=self.random_state)

            query_indices_x, utilities_x = gs.query(X_cand, X=X[~np.isnan(y)],
                                                    batch_size=batch_size_x,
                                                    return_utilities=True)
            query_indices[0:batch_size_x] = query_indices_x
            utilities[0:batch_size_x, :] = utilities_x
        if batch_size_y > 0:
            gs = GSx(x_metric=self.y_metric, random_state=self.random_state)

            query_indices_y, utilities_y = gs.query(y_pred, X=y[~np.isnan(y)],
                                                    batch_size=batch_size_y,
                                                    return_utilities=True)
            query_indices[batch_size_x:batch_size] = query_indices_y
            utilities[batch_size_x:batch_size, :] = utilities_y

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
