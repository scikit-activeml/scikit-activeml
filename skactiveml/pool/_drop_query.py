"""
Module implementing the pool-based query strategy `DropQuery`.
"""

import numpy as np
from sklearn import clone
from sklearn.cluster import KMeans

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_type,
    check_equal_missing_label,
    rand_argmax,
    check_scalar,
)


class DropQuery(SingleAnnotatorPoolQueryStrategy):
    """Dropout Query (DropQuery)

    This class implements  the query strategy Dropout Query (DropQuery) [1]_
    that incorporates both uncertainty and sample diversity into every selected
    batch. For this purpose, unlabeled samples are filtered according to a
    disagreement-based measure via dropout such that only the unlabeled samples
    with a disagreement above a threshold are clustered for selecting the
    unlabeled samples nearest to the respective clusters.

    Parameters
    ----------
    dropout_rate : float, default=0.75
        Dropout rate used to generate samples.
    n_dropout_samples : int, default=3
        Number of dropout samples.
    cluster_algo : ClusterMixin.__class__, default=KMeans
        The cluster algorithm to be used. It must implement a `fit_transform`
        method, which takes samples `X` as inputs, e.g.,
        `sklearn.clustering.KMeans` and `sklearn.clustering.MiniBatchKMeans`.
    cluster_algo_dict : dict, default=None
        The parameters passed to the clustering algorithm `cluster_algo`,
        excluding the parameter for the number of clusters.
    n_cluster_param_name : string, default="n_clusters"
        The name of the parameter for the number of clusters.
    clf_embedding_flag_name : dict or str or None, default=None
        Flag, which is passed to the `predict` method for
        getting the (learned) sample representations.

        - If `clf_embedding_flag_name is None` and `predict` returns
          only one output, the input samples `X` are used.
        - If `clf_embedding_flag_name is None` and `predict` returns
          two outputs, `(y_pred, embeddings)` are expected as outputs.
        - If `isinstance(clf_embedding_name, str)`, we call::

            clf.predict(X, **{clf_embedding_flag_name: True})

          and expect `(y_pred, embeddings)` as output.
        - If `isinstance(clf_embedding_name, dict)`, we call::

            clf.predict(X, **clf_embedding_flag_name)

          and expect `(y_pred, embeddings)` as output.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.
    multilabel_aggregation_fn: callable, default=np.average
        Callable that takes axis as kwarg and reduces along that axis. Common
        choices are `np.mean`, `np.min`, `np.max`, or any quantiles, while
        `np.sum` is not allowed.


    References
    ----------
    .. [1] S. R. Gupte, J. Aklilu, J. J. Nirschl, and S. Yeung-Levy,
       "Revisiting Active Learning in the Era of Vision Foundation Models."
       Trans. Mach. Learn., 2024.
    """

    def __init__(
        self,
        dropout_rate=0.75,
        n_dropout_samples=5,
        cluster_algo=KMeans,
        cluster_algo_dict=None,
        n_cluster_param_name="n_clusters",
        clf_embedding_flag_name=None,
        missing_label=MISSING_LABEL,
        random_state=None,
        multilabel_aggregation_fn=np.average,
    ):
        self.dropout_rate = dropout_rate
        self.n_dropout_samples = n_dropout_samples
        self.cluster_algo = cluster_algo
        self.cluster_algo_dict = cluster_algo_dict
        self.n_cluster_param_name = n_cluster_param_name
        self.clf_embedding_flag_name = clf_embedding_flag_name
        self.multilabel_aggregation_fn = multilabel_aggregation_fn
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Query the next samples to be labeled.

        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`). If `y` is two-dimensional, a
            row `y[i]` must be either contain only observed labels or only
            `missing_label` values, i.e., no mixing within a row.
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates,) of type \
                int, default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. The indexing refers to the samples in `X`.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan. The indexing
            refers to the samples in `X`.
        """
        # Check `__init__` and `query` parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            allow_multioutput=True,
        )
        is_multioutput = y.ndim == 2
        X_cand, mapping = self._transform_candidates(
            candidates=candidates,
            X=X,
            y=y,
            enforce_mapping=True,
            is_multioutput=is_multioutput,
        )
        check_scalar(
            self.dropout_rate,
            name="dropout_rate",
            min_val=0.0,
            max_val=1.0,
            min_inclusive=False,
            max_inclusive=False,
            target_type=float,
        )
        check_scalar(
            self.n_dropout_samples,
            name="n_dropout_samples",
            min_val=3,
            min_inclusive=True,
            target_type=int,
        )
        check_type(
            self.cluster_algo_dict, "cluster_algo_dict", (dict, type(None))
        )
        cluster_algo_dict = (
            {}
            if self.cluster_algo_dict is None
            else self.cluster_algo_dict.copy()
        )
        check_type(self.n_cluster_param_name, "n_cluster_param_name", str)
        check_type(clf, "clf", SkactivemlClassifier)
        check_type(fit_clf, "fit_clf", bool)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        predict_proba_kwargs = {}
        if self.clf_embedding_flag_name is not None:
            check_type(
                self.clf_embedding_flag_name,
                "clf_embedding_flag_name",
                dict,
                str,
            )
            if isinstance(self.clf_embedding_flag_name, str):
                predict_proba_kwargs = {self.clf_embedding_flag_name: True}
            else:
                predict_proba_kwargs = self.clf_embedding_flag_name

        # Fit the classifier, if requested.
        if fit_clf:
            if sample_weight is not None:
                clf = clone(clf).fit(X, y, sample_weight)
            else:
                clf = clone(clf).fit(X, y)

        # Compute predictions and optionally embeddings for original samples.
        y_pred = clf.predict(X_cand, **predict_proba_kwargs)
        if isinstance(y_pred, tuple):
            y_pred, X_embed = y_pred
        else:
            X_embed = X_cand

        # Number of candidate samples.
        n_candidates = len(X_cand)

        # Prepare an array to hold the dropout predictions.
        shape = (n_candidates, self.n_dropout_samples)
        if is_multioutput:
            shape = shape + (y.shape[1],)

        y_pred_dropout = np.empty(shape, dtype=object)

        # Loop over the number of dropout inferences.
        for i in range(self.n_dropout_samples):
            # Copy the candidates so as not to modify the original data.
            X_dropout = X_cand.copy()

            # Generate and apply the dropout mask.
            dropout_mask = self.random_state_.choice(
                [True, False],
                size=X_dropout.shape,
                p=[self.dropout_rate, 1 - self.dropout_rate],
            )
            X_dropout[dropout_mask] = 0.0

            # Compute class predictions for this dropout sample.
            y_pred_dropout_current = clf.predict(X_dropout)
            if isinstance(y_pred_dropout_current, tuple):
                y_pred_dropout_current, _ = y_pred_dropout_current
            y_pred_dropout[:, i] = y_pred_dropout_current

        # Filter candidates for clustering based on disagreement.
        n_disagrees = (y_pred[:, None, :] != y_pred_dropout).sum(axis=1)
        if is_multioutput:
            n_disagrees = self.multilabel_aggregation_fn(n_disagrees, axis=-1)
        disagree_rate = n_disagrees.astype(float) / self.n_dropout_samples
        n_threshold_samples = max(((disagree_rate > 0.5).sum(), batch_size))
        prefiltered_indices = np.argsort(disagree_rate)[-n_threshold_samples:]

        # Perform clustering to get centroids.
        cluster_algo_dict[self.n_cluster_param_name] = batch_size
        cluster_obj = self.cluster_algo(**cluster_algo_dict)
        dist = cluster_obj.fit_transform(X_embed[prefiltered_indices], y=None)

        # Determine `query_indices` of the samples being closest to the
        # respective centroids.
        query_indices = []
        utilities = np.full((batch_size, len(X)), fill_value=np.nan)
        for b in range(batch_size):
            utilities[b][mapping] = -np.inf
            utilities[b][mapping[prefiltered_indices]] = -dist[:, b]
            utilities[b][query_indices] = np.nan
            idx_b = rand_argmax(utilities[b], random_state=self.random_state_)
            query_indices.append(idx_b[0])
        query_indices = np.array(query_indices, dtype=int)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
