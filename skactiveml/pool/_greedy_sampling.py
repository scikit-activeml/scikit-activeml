import numpy as np
from sklearn import clone
from sklearn.metrics import pairwise_distances, pairwise

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
    """Greedy Sampling in the Feature Space (GSx)

    This class implements the query strategy Greedy Sampling in the Feature
    Space (GSx) [1]_ that tries to select those samples that increase the
    diversity of the feature space the most.

    Parameters
    ----------
    metric : str, default="euclidean"
        Metric used for calculating the distances of the samples in the feature
        space. It must be a valid argument for
        `sklearn.metrics.pairwise_distances` argument `metric`.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the pairwise_distances
        function.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] D. Wu, C.-T. Lin, and J. Huang. Active Learning for Regression using
       Greedy Sampling. Inf. Sci., 474:90–105, 2019.
    """

    def __init__(
        self,
        metric=None,
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.metric = metric
        self.metric_dict = metric_dict

    def query(
        self, X, y, candidates=None, batch_size=1, return_utilities=False
    ):
        """Query the next samples to be labeled.

        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`.)
        candidates : None or array-like of shape (n_candidates, ) of type \
                int, default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, `candidates` is
              considered as the candidate samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is to
            be queried, e.g., `query_indices[0]` indicates the first selected
            sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None`, the indexing refers to the samples
              in `X`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `utilities` refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, *)`, `utilities`
              refers to the indexing in `candidates`.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        sample_indices = np.arange(len(X), dtype=int)
        selected_indices = labeled_indices(y, missing_label=self.missing_label)

        if mapping is None:
            X_all = np.append(X, X_cand, axis=0)
            candidate_indices = len(X) + np.arange(len(X_cand), dtype=int)
        else:
            X_all = X
            candidate_indices = mapping

        query_indices_cand, utilities_cand = _greedy_sampling(
            X_cand,
            X_all,
            sample_indices,
            selected_indices,
            candidate_indices,
            batch_size,
            random_state=self.random_state_,
            method="x",
            metric_x=self.metric,
            metric_dict_x=self.metric_dict,
        )

        if mapping is not None:
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand
            query_indices = mapping[query_indices_cand]
        else:
            utilities, query_indices = utilities_cand, query_indices_cand

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


class GreedySamplingTarget(SingleAnnotatorPoolQueryStrategy):
    """Greedy Sampling in the Target Space (GSi or GSy)

    This class implements the query strategy Greedy Sampling in the Target
    Space (GSi or GSy) [1]_ that at first selects samples to maximize the
    diversity in the feature space and than selects samples to maximize the
    diversity in the feature and the target space (GSi), optionally only the
    diversity in the target space can be maximized (GSy).

    Parameters
    ----------
    x_metric : str, default=None
        Metric used for calculating the distances of the samples in the feature
        space. It must be a valid argument for
        `sklearn.metrics.pairwise_distances` argument `metric`.
    y_metric : str, default=None
        Metric used for calculating the distances of the samples in the target
        space. It must be a valid argument for
        `sklearn.metrics.pairwise_distances` argument `metric`.
    x_metric_dict : dict, default=None
        Any further parameters for computing the distances of the samples in
        the feature space are passed directly to the pairwise_distances
        function.
    y_metric_dict : dict, default=None
        Any further parameters for computing the distances of the samples in
        the target space are passed directly to the pairwise_distances
        function.
    n_GSx_samples : int, default=1
        Indicates the number of selected samples required till the query
        strategy switches from GSx to the strategy specified by `method`.
    method : "GSy" or "GSi", optional (default="GSi")
        Specifies whether only the diversity in the target space ("GSy") or the
        diversity in the feature and the target space ("GSi") should be
        maximized, when the number of selected samples exceeds `n_GSx_samples`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] D. Wu, C.-T. Lin, and J. Huang. Active Learning for Regression using
       Greedy Sampling. Inf. Sci., 474:90–105, 2019.
    """

    def __init__(
        self,
        x_metric=None,
        y_metric=None,
        x_metric_dict=None,
        y_metric_dict=None,
        method=None,
        n_GSx_samples=1,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.method = method
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.x_metric_dict = x_metric_dict
        self.y_metric_dict = y_metric_dict
        self.n_GSx_samples = n_GSx_samples

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
        """Query the next samples to be labeled.

        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`.)
        candidates : None or array-like of shape (n_candidates, ) of type \
                int, default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, `candidates` is
              considered as the candidate samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is to
            be queried, e.g., `query_indices[0]` indicates the first selected
            sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None`, the indexing refers to the samples
              in `X`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `utilities` refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, *)`, `utilities`
              refers to the indexing in `candidates`.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        check_type(reg, "reg", SkactivemlRegressor)
        check_type(fit_reg, "fit_reg", bool)
        if self.method is None:
            self.method = "GSi"
        check_type(self.method, "self.method", target_vals=["GSy", "GSi"])
        check_scalar(self.n_GSx_samples, "self.k_0", int, min_val=0)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        n_labeled = np.sum(is_labeled(y, missing_label=self.missing_label_))
        batch_size_x = max(0, min(self.n_GSx_samples - n_labeled, batch_size))
        batch_size_y = batch_size - batch_size_x

        if fit_reg:
            if sample_weight is None:
                reg = clone(reg).fit(X, y)
            else:
                reg = clone(reg).fit(X, y, sample_weight)

        sample_indices = np.arange(len(X), dtype=int)
        selected_indices = labeled_indices(y)
        y_cand = reg.predict(X_cand)

        if mapping is None:
            X_all = np.append(X, X_cand, axis=0)
            y_all = np.append(y, reg.predict(X_cand))
            candidate_indices = len(X) + np.arange(len(X_cand), dtype=int)
        else:
            X_all = X
            y_all = y.copy()
            y_all[mapping] = y_cand
            candidate_indices = mapping

        query_indices = np.zeros(batch_size, dtype=int)
        utilities = np.full((batch_size, len(X_cand)), np.nan)

        if batch_size_x > 0:
            query_indices_x, utilities_x = _greedy_sampling(
                X_cand=X_cand,
                y_cand=y_cand,
                X=X_all,
                y=y_all,
                sample_indices=sample_indices,
                selected_indices=selected_indices,
                candidate_indices=candidate_indices,
                batch_size=batch_size_x,
                random_state=None,
                metric_x=self.x_metric,
                metric_dict_x=self.x_metric_dict,
                method="x",
            )

            query_indices[0:batch_size_x] = query_indices_x
            utilities[0:batch_size_x, :] = utilities_x

        else:
            query_indices_x = np.array([], dtype=int)

        selected_indices = np.append(
            selected_indices, candidate_indices[query_indices_x]
        )
        candidate_indices = np.delete(candidate_indices, query_indices_x)
        is_queried = np.full(len(X_cand), False)
        is_queried[query_indices_x] = True
        unselected_cands = np.argwhere(~is_queried).flatten()

        X_cand = np.delete(X_cand, query_indices_x, axis=0)
        y_cand = np.delete(y_cand, query_indices_x)

        if batch_size_y > 0:
            query_indices_y, utilities_y = _greedy_sampling(
                X_cand=X_cand,
                y_cand=y_cand,
                X=X_all,
                y=y_all,
                sample_indices=sample_indices,
                selected_indices=selected_indices,
                candidate_indices=candidate_indices,
                batch_size=batch_size_y,
                random_state=None,
                metric_x=self.x_metric,
                metric_dict_x=self.x_metric_dict,
                metric_y=self.y_metric,
                metric_dict_y=self.y_metric_dict,
                method="xy" if self.method == "GSi" else "y",
            )

            query_indices[batch_size_x:] = unselected_cands[query_indices_y]
            utilities[batch_size_x:][:, unselected_cands] = utilities_y

        if mapping is not None:
            utilities_cand, query_indices_cand = utilities, query_indices
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand
            query_indices = mapping[query_indices_cand]

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


def _greedy_sampling(
    X_cand,
    X,
    sample_indices,
    selected_indices,
    candidate_indices,
    batch_size,
    y_cand=None,
    y=None,
    random_state=None,
    method=None,
    **kwargs,
):
    dist_dict = dict(
        X_cand=X_cand, y_cand=y_cand, X=X, y=y, method=method, **kwargs
    )
    query_indices = np.zeros(batch_size, dtype=int)
    utilities = np.full((batch_size, len(X_cand)), np.nan)
    distances = np.full((len(X_cand), len(X)), np.nan)

    if len(selected_indices) == 0:
        distances[:, sample_indices] = _measure_distance(
            sample_indices, **dist_dict
        )
    else:
        distances[:, selected_indices] = _measure_distance(
            selected_indices, **dist_dict
        )

    not_selected_candidates = np.arange(len(X_cand), dtype=int)

    for i in range(batch_size):
        if len(selected_indices) == 0:
            dist = distances[not_selected_candidates][:, sample_indices]
            util = -np.sum(dist, axis=1)
        else:
            dist = distances[not_selected_candidates][:, selected_indices]
            util = np.min(dist, axis=1)
        utilities[i, not_selected_candidates] = util

        idx = rand_argmax(util, random_state=random_state)
        query_indices[i] = not_selected_candidates[idx][0]
        distances[:, candidate_indices[idx]] = _measure_distance(
            candidate_indices[idx], **dist_dict
        )

        selected_indices = np.append(
            selected_indices, candidate_indices[idx], axis=0
        )
        candidate_indices = np.delete(candidate_indices, idx, axis=0)
        not_selected_candidates = np.delete(not_selected_candidates, idx)

    return query_indices, utilities


def _measure_distance(
    indices,
    X_cand,
    y_cand,
    X,
    y,
    metric_dict_x=None,
    metric_x=None,
    metric_dict_y=None,
    metric_y=None,
    method=None,
):
    metric_x = metric_x if metric_x is not None else "euclidean"
    metric_y = metric_y if metric_y is not None else "euclidean"

    for metric, name in zip([metric_x, metric_y], ["metric_x", "metric_y"]):
        check_type(
            metric,
            name,
            target_vals=pairwise.PAIRWISE_DISTANCE_FUNCTIONS.keys(),
        )

    metric_dict_x = metric_dict_x if metric_dict_x is not None else {}
    metric_dict_y = metric_dict_y if metric_dict_y is not None else {}

    for metric_dict, name in zip(
        [metric_dict_x, metric_dict_y], ["metric_dict_x", "metric_dict_y"]
    ):
        check_type(metric_dict, name, dict)

    dist = np.ones((len(X_cand), len(indices)))

    if "x" in method:
        dist *= pairwise_distances(
            X_cand, X[indices], metric=metric_x, **metric_dict_x
        )
    if "y" in method:
        dist *= pairwise_distances(
            y_cand.reshape(-1, 1),
            y[indices].reshape(-1, 1),
            metric=metric_y,
            **metric_dict_y,
        )
    return dist
