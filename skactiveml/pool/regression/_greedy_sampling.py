import numpy as np
from sklearn import clone
from sklearn.metrics import pairwise_distances

from skactiveml.base import (
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlRegressor,
)
from skactiveml.utils import (
    rand_argmax,
    labeled_indices,
    MISSING_LABEL,
    is_labeled,
    check_type,
    check_scalar,
)


class GreedySamplingX(SingleAnnotatorPoolQueryStrategy):
    """Greedy Sampling on the feature space

    This class implements greedy sampling on the feature space.

    Parameters
    ----------
    metric: str, optional (default=None)
        Metric used for calculating the distances of points in the feature
        space must be a valid argument for `sklearn.metrics.pairwise_distances`
        argument `metric`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Wu, Dongrui, Chin-Teng Lin, and Jian Huang. Active learning for
    regression using greedy sampling, pages 90--105, 2019.

    """

    def __init__(
        self,
        metric=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(random_state=random_state, missing_label=missing_label)
        self.metric = metric if metric is not None else "euclidean"

    def query(
        self, X, y, candidates=None, batch_size=1, return_utilities=False
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

        query_indices = np.zeros(batch_size, dtype=int)
        is_sample = np.arange(len(X), dtype=int)

        if mapping is None:
            X_all = np.append(X, X_cand, axis=0)
            selected_indices = labeled_indices(y)
            candidate_indices = len(X) + np.arange(len(X_cand), dtype=int)
        else:
            X_all = X
            selected_indices = labeled_indices(y)
            candidate_indices = mapping

        utilities = np.full((batch_size, len(X_all)), np.nan)
        distances = pairwise_distances(X_all, metric=self.metric)

        for i in range(batch_size):
            if selected_indices.shape[0] == 0:
                dist = distances[candidate_indices][:, is_sample]
                util = -np.sum(dist, axis=1)
            else:
                dist = distances[candidate_indices][:, selected_indices]
                util = np.min(dist, axis=1)
            utilities[i, candidate_indices] = util

            idx = rand_argmax(util, random_state=self.random_state)
            query_indices[i] = candidate_indices[idx]
            selected_indices = np.append(
                selected_indices, candidate_indices[idx], axis=0
            )
            candidate_indices = np.delete(candidate_indices, idx, axis=0)

        if mapping is None:
            query_indices = query_indices - len(X)
            utilities = utilities[:, len(X) :]

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


class GreedySamplingY(SingleAnnotatorPoolQueryStrategy):
    """Greedy Sampling on the target space

    This class implements greedy sampling on the target space.

    Parameters
    ----------
    x_metric: str, optional (default=None)
        Metric used for calculating the distances of points in the feature
        space must be a valid argument for `sklearn.metrics.pairwise_distances`
        argument `metric`.
    y_metric: str, optional (default=None)
        Metric used for calculating the distances of points in the target
        space must be a valid argument for `sklearn.metrics.pairwise_distances`
        argument `metric`.
    k_0: int, optional (default=1)
        The minimum number of samples the estimator requires.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Wu, Dongrui, Chin-Teng Lin, and Jian Huang. Active learning for
    regression using greedy sampling, pages 90--105, 2019.

    """

    def __init__(
        self,
        x_metric="euclidean",
        y_metric="euclidean",
        k_0=1,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(random_state=random_state, missing_label=missing_label)
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.k_0 = k_0

    def query(
        self,
        X,
        y,
        reg,
        fit_reg=True,
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
        fit_reg : bool, optional (default=True)
            Defines whether the regressor should be fitted on `X`, `y`, and
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
        check_type(fit_reg, "fit_reg", bool)
        check_scalar(self.k_0, "self.k_0", int, min_val=1)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_reg:
            reg = clone(reg).fit(X, y, sample_weight)

        n_labeled = np.sum(is_labeled(y, missing_label=self.missing_label_))
        batch_size_x = max(0, min(self.k_0 - n_labeled, batch_size))
        batch_size_y = batch_size - batch_size_x

        query_indices = np.zeros((batch_size,), dtype=int)

        if mapping is None:
            utilities = np.full((batch_size, len(X_cand)), np.nan)
        else:
            utilities = np.full((batch_size, len(X)), np.nan)

        if batch_size_x > 0:
            gs = GreedySamplingX(
                metric=self.x_metric, random_state=self.random_state
            )
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

            gs = GreedySamplingX(
                metric=self.y_metric, random_state=self.random_state
            )
            query_indices_y, utilities_y = gs.query(
                # left missing_values are not used, so replace by zero
                X=np.where(is_labeled(y_to_X), y_to_X, 0).reshape(-1, 1),
                y=y_new,
                candidates=y_candidate,
                batch_size=int(batch_size_y),
                return_utilities=True,
            )

            if mapping is None:
                query_indices[batch_size_x:batch_size] = indices_nq[
                    query_indices_y
                ]
                utilities[batch_size_x:batch_size][:, indices_nq] = utilities_y
            else:
                query_indices[batch_size_x:batch_size] = query_indices_y
                utilities[batch_size_x:batch_size] = utilities_y

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
