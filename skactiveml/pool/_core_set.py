import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    MISSING_LABEL,
    labeled_indices,
    unlabeled_indices,
    rand_argmax,
)
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_random_state,
    column_or_1d,
)
from sklearn.metrics import pairwise_distances_argmin_min


class CoreSet(SingleAnnotatorPoolQueryStrategy):
    """Core Set

    This class implements the core-set based query strategy, i.e., the
    standard greedy algorithm for the k-center problem [1]_. CoreSet
    applies a k-center (farthest-first) selection in a feature space,
    seeded by the labeled set, to minimize the maximum distance of unlabeled
    samples to the labeled/selected samples. It is a pure diversity criterion
    without explicit consideration of prediction uncertainty. Originally,
    this query strategy was only proposed for classification. Nevertheless, it
    is task-agnostic such that it can handle class, numerical, and multioutput
    labels.

    Parameters
    ----------
    metric : str or callable, default="euclidean"
        The metric must comply with the function
        `sklearn.metrics.pairwise_distances_argmin_min`.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the function
        `sklearn.metrics.pairwise_distances_argmin_min` as `metric_kwargs`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] O. Sener and S. Savarese. Active Learning for Convolutional Neural
       Networks: A Core-Set Approach. In Int. Conf. Learn. Represent., 2018.
    """

    def __init__(
        self,
        metric="euclidean",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.metric = metric
        self.metric_dict = metric_dict

    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`). If `y` is two-dimensional, a
            row `y[i]` must be either contain only observed labels or only
            `missing_label` values, i.e., no mixing within a row.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """
        # Validate parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
            reset=True,
            allow_multioutput=True,
        )

        # Determine candidate samples for selection.
        is_multioutput = y.ndim == 2
        X_cand, mapping = self._transform_candidates(
            candidates=candidates, X=X, y=y, is_multioutput=is_multioutput
        )

        # Perform selection depending on whether the candidate samples are part
        # of `X` or not.
        if mapping is not None:
            query_indices, utilities = k_greedy_center(
                X=X,
                y=y,
                batch_size=batch_size,
                random_state=self.random_state_,
                missing_label=self.missing_label_,
                mapping=mapping,
                metric=self.metric,
                metric_dict=self.metric_dict,
            )
        else:
            selected_samples = labeled_indices(
                y=y,
                missing_label=self.missing_label_,
                is_multioutput=is_multioutput,
            )
            X_with_cand = np.concatenate((X_cand, X[selected_samples]), axis=0)
            n_new_cand = X_cand.shape[0]
            y_cand = np.full(shape=n_new_cand, fill_value=self.missing_label)
            y_with_cand = np.concatenate(
                (y_cand, y[selected_samples]), axis=None
            )
            mapping = np.arange(n_new_cand)
            query_indices, utilities = k_greedy_center(
                X=X_with_cand,
                y=y_with_cand,
                batch_size=batch_size,
                random_state=self.random_state_,
                missing_label=self.missing_label_,
                mapping=mapping,
                n_new_cand=n_new_cand,
                metric=self.metric,
                metric_dict=self.metric_dict,
                is_multioutput=is_multioutput,
            )
        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


def k_greedy_center(
    X,
    y,
    batch_size=1,
    random_state=None,
    missing_label=MISSING_LABEL,
    mapping=None,
    n_new_cand=None,
    metric="euclidean",
    metric_dict=None,
    is_multioutput=False,
):
    """An active learning method that greedily forms a batch to minimize the
    maximum distance to a cluster center among all unlabeled datapoints.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `self.missing_label`). If `y` is two-dimensional, a
        row `y[i]` must be either contain only observed labels or only
        `missing_label` values, i.e., no mixing within a row.
    batch_size : int, default=1
        The number of samples to be selected in one AL cycle.
    random_state : None or int or np.random.RandomState, default=None
        Random state for candidate selection.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    mapping : None or np.ndarray of shape (n_candidates,), default=None
        Index array that maps `candidates` to `X` (`candidates = X[mapping]`).
    n_new_cand : int or None, default=None
        The number of new candidates that are additionally added to `X`.
        Only used for the case, that in the query function with the shape of
        `candidates` is `(n_candidates, n_feature)`.
    metric : str or callable, default='euclidean'
        The metric must comply with the function
        `sklearn.metrics.pairwise_distances_argmin_min`.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the function
        `sklearn.metrics.pairwise_distances_argmin_min` as `metric_kwargs`.
    is_multioutput : bool, default=False
        Whether provided data is in multioutput format or not.

    Returns
    -------
    query_indices : numpy.ndarray of shape (batch_size)
        The query_indices indicate for which candidate sample a label is
        to queried, e.g., `query_indices[0]` indicates the first selected
        sample.

        - If `candidates` is `None` or of shape
          `(n_candidates,)`, the indexing refers to the samples in
          `X`.
        - If `candidates` is of shape `(n_candidates, n_features)`,
          the indexing refers to the samples in `candidates`.
    utilities : numpy.ndarray of shape (batch_size, n_samples) or \
            numpy.ndarray of shape (batch_size, n_candidates)
        The utilities of samples after each selected sample of the batch,
        e.g., `utilities[0]` indicates the utilities used for selecting
        the first sample (with index `query_indices[0]`) of the batch.
        Utilities for labeled samples will be set to np.nan.

        - If `candidates` is `None` or of shape
          `(n_candidates,)`, the indexing refers to the samples in
          `X`.
        - If `candidates` is of shape `(n_candidates, n_features)`,
          the indexing refers to the samples in `candidates`.
    """
    # Validate input parameters.
    X = check_array(X, allow_nd=True)
    if not is_multioutput:
        y = check_array(
            y, ensure_2d=False, ensure_all_finite="allow-nan", dtype=None
        )
        y = column_or_1d(y, warn=True)
    else:
        y = check_array(
            y, ensure_2d=True, ensure_all_finite="allow-nan", dtype=None
        )
    check_consistent_length(X, y)
    random_state_ = check_random_state(random_state)

    if mapping is None:
        mapping = unlabeled_indices(
            y=y, missing_label=missing_label, is_multioutput=is_multioutput
        )
    else:
        mapping = column_or_1d(mapping, dtype=int, warn=True)

    if not isinstance(batch_size, int):
        raise TypeError("`batch_size` must be an `integer`.")

    # Initialize the utility matrix.
    if n_new_cand is None:
        utilities = np.zeros(shape=(batch_size, X.shape[0]))
    elif isinstance(n_new_cand, int):
        if n_new_cand == len(mapping):
            utilities = np.zeros(shape=(batch_size, n_new_cand))
        else:
            raise ValueError(
                "n_new_cand must equal to the length of mapping array"
            )
    else:
        raise TypeError("Only n_new_cand with type int is supported.")

    selected_samples = labeled_indices(
        y=y, missing_label=missing_label, is_multioutput=is_multioutput
    )
    query_indices = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        if i == 0:
            update_dist = _update_distances(
                X=X,
                cluster_centers=selected_samples,
                mapping=mapping,
                metric=metric,
                metric_dict=metric_dict,
            )
        else:
            latest_dist = utilities[i - 1]
            update_dist = _update_distances(
                X=X,
                cluster_centers=[query_indices[i - 1]],
                mapping=mapping,
                latest_distance=latest_dist,
                metric=metric,
                metric_dict=metric_dict,
            )

        if n_new_cand is None:
            utilities[i] = update_dist
        else:
            utilities[i] = update_dist[mapping]

        # select index
        query_indices[i] = rand_argmax(
            utilities[i], random_state=random_state_
        ).item()

    return query_indices, utilities


def _update_distances(
    X,
    cluster_centers,
    mapping,
    latest_distance=None,
    metric="euclidean",
    metric_dict=None,
):
    """
    Update minimum distances by given cluster centers.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    cluster_centers : array-like of shape (n_cluster_centers)
        Indices of cluster centers.
    mapping : np.ndarray of shape (n_candidates, ), default=None
        Index array that maps `candidates` to `X` (`candidates = X[mapping]`).
    latest_distance : array-like of shape (n_samples) default None
        The distance between each sample and its nearest center. Used to
        speed up the computation of distances for the next selected sample.
    metric : str or callable, default='euclidean'
        The metric must comply with the function
        `sklearn.metrics.pairwise_distances_argmin_min`.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the function
        `sklearn.metrics.pairwise_distances_argmin_min` as `metric_kwargs`.

    Returns
    -------
    result-dist : np.ndarray of shape (1, n_samples)
        - If there aren't any cluster centers existing, the default distance
        will be 0.
        - If there are some cluster center exist, the return will be the
        distance between each sample and its nearest center after each selected
        sample of the batch. In the case of cluster center the value will be
        `np.nan`.
        - For the case, that indices aren't in `mapping`, the corresponding
        value in `result-dist` will be also `np.nan`.
    """
    dist = np.zeros(shape=X.shape[0])
    if metric_dict is None:
        metric_dict = {}

    if len(cluster_centers) > 0:
        cluster_center_feature = X[cluster_centers]
        _, dist = pairwise_distances_argmin_min(
            X, cluster_center_feature, metric=metric, metric_kwargs=metric_dict
        )

    if latest_distance is not None:
        sum_dist = np.nansum(latest_distance)
        latest_distance_tmp = latest_distance
        if sum_dist == 0:
            latest_distance_tmp = latest_distance.copy()
            latest_distance_tmp[latest_distance_tmp == 0] = np.inf
        l_distance = np.zeros(shape=X.shape[0])
        l_distance[mapping] = latest_distance_tmp[mapping]
        dist = np.minimum(l_distance, dist)

    result_dist = np.full(X.shape[0], np.nan)
    result_dist[mapping] = dist[mapping]
    result_dist[cluster_centers] = np.nan

    return result_dist
