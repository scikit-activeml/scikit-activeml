import numpy as np
from scipy import stats
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
    Badge query strategy

    This class implements the Batch Active Learning by Diverse
    Gradient Embedding (BADGE) algorithm from [1]. This approach is
    designed to incorporate both predictive uncertainty and
    sample diversity into every selected batch.

    Parameters
    ----------
    missing_label: scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label
    random_state: int or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal,
    ‘Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds’.
    arXiv, Feb. 23, 2020. Accessed: Dec. 05, 2023. [Online].
    Available: http://arxiv.org/abs/1906.03671
    """

    def __init__(self, missing_label=MISSING_LABEL, random_state=None):
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
        return_embeddings=False,
    ):
        """
        Query strategy for BADGE

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
        y: array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label)
        clf: skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        candidates: None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size: int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities: bool, default=False
            If true, also return the utilities based on the query strategy.
        return_embeddings: bool, default=False
            If true, by the gradient embedding step will also return the
            X_embedding for the calculation of g_x.

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
            The utilities of samples before each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            For the case where sample choose uniform random from the set, the
            utility of all samples will be 1.
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
        check_scalar(return_embeddings, "return_embeddings", bool)

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
        if return_embeddings:
            probas, X_unlbld = clf.predict_proba(X_unlbld, return_embeddings=True)
        else:
            probas = clf.predict_proba(X_unlbld)
        p_max = np.max(probas, axis=1).reshape(-1, 1)
        g_x = (p_max - 1) * X_unlbld

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
        query_indicies = np.array([], dtype=int)
        query_indicies_in_unlbld = np.array([], dtype=int)
        d_2_s = []
        for i in range(batch_size):
            if i == 0:
                d_2 = _d_2(g_x, [])
            else:
                d_2 = _d_2(g_x, [idx_in_unlbld], d_2_s[i - 1])
            d_2_s.append(d_2)

            d_2_sum = np.sum(d_2)
            if i == 0 or d_2_sum == 0:
                if d_2_sum == 0:
                    d_2_s[-1] = np.full(shape=len(g_x), fill_value=np.inf)
                d_2 = np.ones(shape=len(g_x))
                d_2[query_indicies_in_unlbld] = 0
                d_2_sum = np.sum(d_2)

            d_probas = d_2 / d_2_sum

            utilities[i, unlbld_mapping] = d_probas
            utilities[i, query_indicies] = np.nan

            customDist = stats.rv_discrete(
                name="customDist", values=(np.arange(len(d_probas)), d_probas)
            )
            idx_in_unlbld_array = customDist.rvs(
                size=1, random_state=self.random_state_
            )
            # TODO: with self.random_state_.choice(p=d_probas)
            idx_in_unlbld = idx_in_unlbld_array[0]
            query_indicies_in_unlbld = np.append(
                query_indicies_in_unlbld, idx_in_unlbld_array
            )

            idx = unlbld_mapping[idx_in_unlbld]
            query_indicies = np.append(query_indicies, [idx])

        if return_utilities:
            return query_indicies, utilities
        else:
            return query_indicies


def _d_2(g_x, query_indicies, d_latest=None):
    """
    Calculates the D^2 value of the embedding features of unlabeled
    data.

    Parameters
    ----------
    g_x: numpy.ndarray of shape (n_unlabeled_samples, n_features)
        The results after gradient embedding
    query_indicies: numpy.ndarray of shape (n_query_indicies)
        the query indicies, which mapping in the unlabeled samples
    d_latest: numpy.ndarray of shape (n_unlabeled_samples) default=None
        The distance between each data point and its nearest center
        Using to facilitate the computation the later distances for the
        coming selected sample

    Returns
    -------
    D2: numpy.ndarray of shape (n_unlabeled_samples)
        The D^2 value, for the first sample is the value inf.
    """
    if len(query_indicies) == 0:
        return np.full(shape=len(g_x), fill_value=np.inf)
    g_query_indicies = g_x[query_indicies]
    _, D = pairwise_distances_argmin_min(X=g_x, Y=g_query_indicies)
    if d_latest is not None:
        D = np.minimum(d_latest, D)
    D2 = np.square(D)
    return D2
