"""
Module implementing Clustering Uncertainty-weighted Embeddings (CLUE).

CLUE is a deep active learning strategy, which performs a clustering with
uncertainties as sample weights.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..pool import uncertainty_scores
from ..utils import (
    MISSING_LABEL,
    rand_argmax,
    check_type,
    check_equal_missing_label,
)
from sklearn.base import clone
from sklearn.cluster import KMeans


class Clue(SingleAnnotatorPoolQueryStrategy):
    """Clustering Uncertainty-weighted Embeddings (CLUE)

    This class implements the Clustering Uncertainty-weighted Embeddings (CLUE)
    query strategy [1]_, which considers both diversity and uncertainty of the
    samples.

    Parameters
    ----------
    cluster_algo : ClusterMixin.__class__, default=KMeans
        The cluster algorithm to be used. It must implement a `fit_transform`
        method, which takes samples `X` and `sample_weight` as inputs, e.g.,
        sklearn.clustering.KMeans and sklearn.clustering.MiniBatchKMeans.
    cluster_algo_dict : dict, default=None
        The parameters passed to the clustering algorithm `cluster_algo`,
        excluding the parameter for the number of clusters.
    n_cluster_param_name : string, default="n_clusters"
        The name of the parameter for the number of clusters.
    method : 'least_confident' or 'margin_sampling' or 'entropy', \
            default="entropy"
        - `method='least_confident'` queries the sample whose maximal posterior
          probability is minimal.
        - `method='margin_sampling'` queries the sample whose posterior
          probability gap between the most and the second most probable class
          label is minimal.
        - `method='entropy'` queries the sample whose posterior's have the
          maximal entropy.
    clf_embedding_flag_name : str or None, default=None
        Name of the flag, which is passed to the `predict_proba` method for
        getting the (learned) sample representations.

        - If `clf_embedding_flag_name=None` and `predict_proba` returns
          only one output, the input samples `X` are used.
        - If `predict_proba` returns two outputs or `clf_embedding_name` is
          not `None`, `(proba, embeddings)` are expected as outputs.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] V. Prabhu, A. Chandrasekaran, K. Saenko, and J. Hoffman. Active
       domain adaptation via clustering uncertainty-weighted embeddings. In
       IEEE/CVF Int. Conf. Comput. Vis., pages 8505–8514, 2021.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        cluster_algo=KMeans,
        cluster_algo_dict=None,
        n_cluster_param_name="n_clusters",
        method="entropy",
        clf_embedding_flag_name=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.cluster_algo = cluster_algo
        self.cluster_algo_dict = cluster_algo_dict
        self.n_cluster_param_name = n_cluster_param_name
        self.method = method
        self.clf_embedding_flag_name = clf_embedding_flag_name

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
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight: array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates,), dtype=int or\
                array-like of shape (n_candidates, n_features), default=None
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
        # Check `__init__` and `query` parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        X_cand, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=True
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

        # Fit the classifier.
        if fit_clf:
            if sample_weight is not None:
                clf = clone(clf).fit(X, y, sample_weight)
            else:
                clf = clone(clf).fit(X, y)

        # Compute class-membership predictions and optionally embeddings.
        if self.clf_embedding_flag_name is not None:
            probas, X_cand = clf.predict_proba(
                X_cand, **{self.clf_embedding_flag_name: True}
            )
        else:
            probas = clf.predict_proba(X_cand)
            if isinstance(probas, tuple):
                probas, X_cand = probas

        # Compute uncertainties according to given `method`.
        uncertainties = uncertainty_scores(probas=probas, method=self.method)

        # Implement a fallback, if all uncertainties are zero.
        if np.sum(uncertainties) == 0:
            uncertainties = np.ones_like(uncertainties)

        # Perform clustering to get centroids.
        cluster_algo_dict[self.n_cluster_param_name] = batch_size
        cluster_obj = self.cluster_algo(**cluster_algo_dict)
        dist = cluster_obj.fit_transform(
            X_cand, y=None, sample_weight=uncertainties
        )

        # Determine `query_indices` of the samples being closest to the
        # respective centroids.
        query_indices = []
        utilities = np.full((batch_size, len(X)), fill_value=np.nan)
        for b in range(batch_size):
            utilities[b][mapping] = -dist[:, b]
            utilities[b][query_indices] = np.nan
            idx_b = rand_argmax(utilities[b], random_state=self.random_state_)
            query_indices.append(idx_b[0])
        query_indices = np.array(query_indices, dtype=int)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
