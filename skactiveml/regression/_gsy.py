import numpy as np
from sklearn import clone

from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlRegressor
from skactiveml.regression._gsx import GSx
from skactiveml.utils import check_type


class GSy(SingleAnnotatorPoolQueryStrategy):
    """Greedy Sampling on the feature space

    This class implements greedy sampling on the target space.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_0: int, optional (default=1)
        The minimum number of samples the estimator requires.
    """

    def __init__(
        self, x_metric="euclidean", y_metric="euclidean", k_0=1, random_state=None
    ):
        super().__init__(random_state=random_state)
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.k_0 = k_0

    def query(
        self,
        X,
        y,
        reg,
        fit_clf=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        reg: SkactivemlRegressor
            Regressor to predict the data.
        fit_clf : bool, optional (default=True)
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

        check_type(reg, "reg", SkactivemlRegressor)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_clf:
            reg = clone(reg).fit(X, y, sample_weight)

        n_train_samples = X.shape[0]
        y_pred = reg.predict(X_cand)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        is_labeled = np.all(~np.isnan(y), axis=1)

        n_samples = X_cand.shape[0]

        batch_size_x = max(0, min(self.k_0 - n_train_samples, batch_size))
        batch_size_y = batch_size - batch_size_x

        query_indices = np.zeros((batch_size,))
        utilities = np.zeros((batch_size, n_samples))

        if batch_size_x > 0:
            gs = GSx(x_metric=self.x_metric, random_state=self.random_state)

            query_indices_x, utilities_x = gs.query(
                X=X[is_labeled],
                y=y[is_labeled],
                candidates=X_cand,
                batch_size=batch_size_x,
                return_utilities=True,
            )

            query_indices[0:batch_size_x] = query_indices_x
            utilities[0:batch_size_x, :] = utilities_x
        if batch_size_y > 0:
            gs = GSx(x_metric=self.y_metric, random_state=self.random_state)

            query_indices_y, utilities_y = gs.query(
                X=y[is_labeled],
                y=y[is_labeled],
                candidates=y_pred,
                batch_size=batch_size_y,
                return_utilities=True,
            )
            query_indices[batch_size_x:batch_size] = query_indices_y
            utilities[batch_size_x:batch_size, :] = utilities_y

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
