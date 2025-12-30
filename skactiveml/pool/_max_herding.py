import numpy as np

from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import normalize

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    MISSING_LABEL,
    rand_argmax,
    check_type,
    is_labeled,
)


class MaxHerding(SingleAnnotatorPoolQueryStrategy):
    """MaxHerding

    This class implements the MaxHerding query strategy [1]_, which greedily
    selects `batch_size` unlabeled samples that most increase a smooth,
    kernel-based coverage objective in embedding space, accounting for the
    already labeled set. The objective promotes representativeness and
    diversity via kernel similarity. Originally, this query strategy was only
    proposed for classification. Originally, this query strategy was only
    proposed for classification tasks. Nevertheless, this implementation is
    task-agnostic such that it can handle class, numerical, and multioutput
    labels.

    Parameters
    ----------
    normalize_samples : bool, default=True
        Flag whether to normalize the samples to have unit length.
    metric : str or callable, default=None
        The metric must be None or a valid kernel as defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`.
    metric_dict : dict, default=None
        Any further parameters that should be passed directly to the kernel
        function `sklearn.metrics.pairwise.pairwise_kernels`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] Bae, Wonho, Junhyug Noh, and Danica J. Sutherland. "Generalized
       Coverage for More Robust Low-Budget Active Learning."
       In Eur. Conf. Comput. Vis. 2024.
    """

    def __init__(
        self,
        normalize_samples=True,
        metric="rbf",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.normalize_samples = normalize_samples
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
        candidates : None or array-like of shape (n_candidates,), dtype=int \
                or array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. The indexing refers to the samples in `X`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan. The indexing
            refers to the samples in `X`.
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
        metric_dict = {} if self.metric_dict is None else self.metric_dict
        check_type(metric_dict, "metric_dict", dict)
        check_type(self.normalize_samples, "normalize_samples", bool)

        # Determine candidate samples for selection.
        is_multioutput = y.ndim == 2
        X_cand, mapping = self._transform_candidates(
            candidates=candidates, X=X, y=y, is_multioutput=is_multioutput
        )

        # Precompute kernel values (cf. line 1 of Algorithm 1 in [1]).
        if self.normalize_samples:
            X_cand = normalize(X_cand, copy=True)
        K_cand = pairwise_kernels(X_cand, metric=self.metric, **metric_dict)
        k_max = None
        is_lbld = is_labeled(
            y=y,
            missing_label=self.missing_label_,
            is_multioutput=is_multioutput,
        )
        if is_lbld.sum() > 0:
            X_lbld = X[is_lbld]
            if self.normalize_samples:
                X_lbld = normalize(X_lbld, copy=True)
            K_cand_labeled = pairwise_kernels(
                X_cand, X_lbld, metric=self.metric, **metric_dict
            )
            k_max = K_cand_labeled.max(axis=1)

        # Storages for saving query indices and utilities.
        query_indices_cand = np.empty(batch_size, dtype=int)
        utilities_cand = np.empty((batch_size, len(X_cand)), dtype=float)

        # Greedy selection (cf. lines 3 to 6 of Algorithm 1 in [1]).
        for b in range(batch_size):
            if k_max is not None:
                # Compute utilities if labeled data is available.
                utilities_cand[b] = np.mean(
                    np.maximum(K_cand - k_max, 0), axis=1
                )
            else:
                # Fallback to the kernel-based densities as utilities if
                # labeled data is unavailable.
                utilities_cand[b] = K_cand.mean(axis=1)
                k_max = np.zeros(len(X_cand), dtype=float)
            utilities_cand[b][query_indices_cand[:b]] = np.nan
            query_indices_cand[b] = rand_argmax(
                utilities_cand[b], random_state=self.random_state_
            )[0]
            k_max = np.maximum(K_cand[:, query_indices_cand[b]], k_max)

        if mapping is None:
            query_indices = query_indices_cand
            utilities = utilities_cand
        else:
            query_indices = mapping[query_indices_cand]
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
