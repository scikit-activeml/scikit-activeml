import numpy as np
from sklearn import clone

from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlRegressor
from skactiveml.pool.regression._greedy_sampling_x import GSx
from skactiveml.utils import check_type, is_labeled, check_scalar


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
        check_scalar(self.k_0, "self.k_0", int, min_val=1)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_clf:
            reg = clone(reg).fit(X, y, sample_weight)

        n_labeled = np.sum(is_labeled(y))
        batch_size_x = max(0, min(self.k_0 - n_labeled, batch_size))
        batch_size_y = batch_size - batch_size_x

        query_indices = np.zeros((batch_size,), dtype=int)

        if mapping is None:
            utilities = np.full((batch_size, len(X_cand)), np.nan)
        else:
            utilities = np.full((batch_size, len(X)), np.nan)

        if batch_size_x > 0:
            gs = GSx(metric=self.x_metric, random_state=self.random_state)
            query_indices_x, utilities_x = gs.query(
                X=X,
                y=y,
                candidates=candidates,
                batch_size=int(batch_size_x),
                return_utilities=True,
            )

            query_indices[0:batch_size_x] = query_indices_x
            utilities[0:batch_size_x, :] = utilities_x
            if mapping is not None:
                query_indices_x = mapping[query_indices_x]

        else:
            query_indices_x = np.zeros(0, dtype=int)

        if batch_size_y > 0:
            is_queried = np.full(len(X_cand), False)
            is_queried[query_indices_x] = True
            # not all ready queried indices
            indices_nq = np.argwhere(~is_queried).flatten()

            y_to_X = y.copy()
            y_pred = reg.predict(X_cand)

            if mapping is None:
                y_to_X = np.append(y, y_pred[is_queried])
                y_new = y_to_X
                y_candidate = y_pred[~is_queried].reshape(-1, 1)
            else:
                y_new = y_to_X.copy()
                y_new[mapping[is_queried]] = y_pred[is_queried]
                y_to_X[mapping] = y_pred
                y_candidate = mapping[~is_queried]

            gs = GSx(metric=self.y_metric, random_state=self.random_state)
            query_indices_y, utilities_y = gs.query(
                # replace missing_value by 0, since it does not implement
                X=np.where(is_labeled(y_to_X), y_to_X, 0).reshape(-1, 1),
                y=y_new,
                candidates=y_candidate,
                batch_size=int(batch_size_y),
                return_utilities=True,
            )

            if mapping is None:
                query_indices[batch_size_x:batch_size] = indices_nq[query_indices_y]
                utilities[batch_size_x:batch_size][:, indices_nq] = utilities_y
            else:
                query_indices[batch_size_x:batch_size] = query_indices_y
                utilities[batch_size_x:batch_size] = utilities_y

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
