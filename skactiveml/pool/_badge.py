"""
Module implementing Batch Active Learning by Diverse Gradient Embedding (BADGE)
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
    """
    Batch Active Learning by Diverse Gradient Embedding (BADGE)

    This class implements the BADGE algorithm [1]. This query strategy is
    designed to incorporate both predictive uncertainty and
    sample diversity into every selected batch.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState
        The random state to use.

    References
    ----------
     [1] J. Ash, Jordan T., Chicheng Zhang, Akshay Krishnamurthy,
         John Langford, and Alekh Agarwal, "Deep Batch Active Learning
         by Diverse, Uncertain Gradient Lower Bounds." ICLR, 2019.
    """

    def __init__(
        self,
        clf_embedding_flag=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        self.clf_embedding_flag = clf_embedding_flag
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Query the next samples to be labeled

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeled samples,
            indicated by self.missing_label).
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
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
        clf_embedding_flag : string, optional (default=None)
            The name of the parameter, whether to return the embedding or not,
            according to the used classifier.
            By default, `None` means no embeddings will be returned, but we use
            the original `X` as embedding.


        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            being queried for a label, e.g., `query_indices[0]` indicates the
            first selected sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples before each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            For the case where the samples are uniformly randomly selected from the set,
            the sum of all utility of samples will be 1.
            The utilities represent here the probabilities of samples being
            chosen.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
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
        if self.clf_embedding_flag is not None:
            check_scalar(self.clf_embedding_flag, "clf_embedding_flag", str)

        # Fit the classifier
        if fit_clf:
            clf = clone(clf).fit(X, y)

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
        if self.clf_embedding_flag is not None:
            probas, X_unlbld = clf.predict_proba(
                X_unlbld, **{self.clf_embedding_flag: True}
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
        d_2_s = []
        for i in range(batch_size):
            if i == 0:
                d_2 = _d_2(g_x, [])
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


def _d_2(g_x, query_indicies, d_latest=None):
    """
    Calculates the D^2 value of the embedding features of unlabeled data.

    Parameters
    ----------
    g_x : numpy.ndarray of shape (n_unlabeled_samples, n_features)
        The results after gradient embedding
    query_indicies : numpy.ndarray of shape (n_query_indicies)
        the query indications that correspond to the unlabeled samples.
    d_latest : numpy.ndarray of shape (n_unlabeled_samples) default=None
        The distance between each data point and its nearest centre.
        This is used to simplify the calculation of the later distances for the
        next selected sample.

    Returns
    -------
    D2 : numpy.ndarray of shape (n_unlabeled_samples)
        The D^2 value, for the first sample, is the value inf.
    """
    if len(query_indicies) == 0:
        return np.sum(g_x**2, axis=-1)
    g_query_indicies = g_x[query_indicies]
    _, D = pairwise_distances_argmin_min(X=g_x, Y=g_query_indicies)
    if d_latest is not None:
        D2 = np.minimum(d_latest, np.square(D))
    return D2
