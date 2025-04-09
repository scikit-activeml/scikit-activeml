"""
Module implementing the pool-based query strategy Batch Active Learning by
Diverse Gradient Embedding (BADGE).
"""

import numpy as np
from sklearn import clone
from sklearn.metrics import pairwise_distances_argmin_min

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_type,
    check_equal_missing_label,
    unlabeled_indices,
    check_scalar,
)


class Badge(SingleAnnotatorPoolQueryStrategy):
    """Batch Active Learning by Diverse Gradient Embedding (BADGE)

    This class implements the BADGE algorithm [1]_, which is designed to
    incorporate both predictive uncertainty and sample diversity into every
    selected batch.

    Parameters
    ----------
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
    .. [1] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal.
       Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds.
       In Int. Conf. Learn. Represent., 2020.
    """

    def __init__(
        self,
        clf_embedding_flag_name=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        self.clf_embedding_flag_name = clf_embedding_flag_name
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
            - If `candidates` is of shape `(n_candidates, *)`, the
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
        # Validate input parameters
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate classifier type
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)
        if self.clf_embedding_flag_name is not None:
            check_scalar(
                self.clf_embedding_flag_name, "clf_embedding_flag_name", str
            )

        # Fit the classifier
        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)

        # find the unlabeled dataset
        if candidates is None:
            X_unlbld = X_cand
            unlbld_mapping = mapping
        elif mapping is not None:
            unlbld_mapping = unlabeled_indices(
                y[mapping], missing_label=self.missing_label
            )
            X_unlbld = X_cand[unlbld_mapping]
            unlbld_mapping = mapping[unlbld_mapping]
        else:
            X_unlbld = X_cand
            unlbld_mapping = np.arange(len(X_cand))

        # gradient embedding, aka predict class membership probabilities
        if self.clf_embedding_flag_name is not None:
            probas, X_unlbld = clf.predict_proba(
                X_unlbld, **{self.clf_embedding_flag_name: True}
            )
        else:
            probas = clf.predict_proba(X_unlbld)
            if isinstance(probas, tuple):
                probas, X_unlbld = probas

        y_pred = probas.argmax(axis=-1)
        proba_factor = probas - np.eye(probas.shape[1])[y_pred]
        g_x = proba_factor[:, :, None] * X_unlbld[:, None, :]
        g_x = g_x.reshape(*g_x.shape[:-2], -1)

        # init the utilities
        if mapping is not None:
            utilities = np.full(
                shape=(batch_size, X.shape[0]), fill_value=np.nan
            )
        else:
            utilities = np.full(
                shape=(batch_size, X_cand.shape[0]), fill_value=np.nan
            )

        # sampling with kmeans++
        query_indicies = []
        query_indicies_in_unlbld = []
        idx_in_unlbld = []
        d_2_s = []
        for i in range(batch_size):
            if i == 0:
                d_2 = _d_2(g_x, idx_in_unlbld)
            else:
                d_2 = _d_2(g_x, [idx_in_unlbld], d_2_s[i - 1])
            d_2_s.append(d_2)

            d_2_sum = np.sum(d_2)
            if d_2_sum == 0:
                d_2_s[-1] = np.full(shape=len(g_x), fill_value=np.inf)
                d_2 = np.ones(shape=len(g_x))
                d_2[query_indicies_in_unlbld] = 0
                d_2_sum = np.sum(d_2)

            d_probas = d_2 / d_2_sum

            utilities[i, unlbld_mapping] = d_probas
            utilities[i, query_indicies] = np.nan

            if i == 0 and d_2_sum != 0:
                idx_in_unlbld = np.argmax(d_2, axis=-1)
            else:
                idx_in_unlbld_array = self.random_state_.choice(
                    len(d_probas), 1, replace=False, p=d_probas
                )
                idx_in_unlbld = idx_in_unlbld_array[0]
            query_indicies_in_unlbld.append(idx_in_unlbld)

            idx = unlbld_mapping[idx_in_unlbld]
            query_indicies.append(idx)

        if return_utilities:
            return query_indicies, utilities
        else:
            return query_indicies


def _d_2(g_x, query_indices, d_latest=None):
    """
    Calculates the D^2 value of the embedding features of unlabeled data.

    Parameters
    ----------
    g_x : np.ndarray of shape (n_unlabeled_samples, n_features)
        The results after gradient embedding
    query_indices : numpy.ndarray of shape (n_query_indices,)
        the query indications that correspond to the unlabeled samples.
    d_latest : np.ndarray of shape (n_unlabeled_samples,) default=None
        The distance between each data point and its nearest centre.
        This is used to simplify the calculation of the later distances for the
        next selected sample.

    Returns
    -------
    D2 : numpy.ndarray of shape (n_unlabeled_samples,)
        The D^2 value, for the first sample, is the value inf.
    """
    if len(query_indices) == 0:
        return np.sum(g_x**2, axis=-1)
    query_indices = g_x[query_indices]
    _, D = pairwise_distances_argmin_min(X=g_x, Y=query_indices)
    if d_latest is not None:
        D2 = np.minimum(d_latest, np.square(D))
    return D2
