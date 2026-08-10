"""
Module implementing the pool-based query strategy Batch Active Learning by
Diverse Gradient Embedding (BADGE).
"""

import warnings

import numpy as np
from sklearn import clone

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

    This class implements the BADGE algorithm [1]_, which selects a batch by
    running k-means++ on per-sample gradient embeddings, which combine
    uncertainty and diversity. For each unlabeled sample, it forms the gradient
    of the cross-entropy loss with respect to the last linear layer using the
    model’s pseudo-label. Large gradient norms indicate uncertainty, while
    k-means++ spreads selections to avoid redundancy.

    The gradient embedding of a sample is the Kronecker product
    `g = kron(q, v)` of its probability residual `q` and its (learned) sample
    representation `v`. Since inner products factorize as
    `<g_i, g_j> = <q_i, q_j> * <v_i, v_j>` [2]_, the
    `(n_samples, n_classes * n_features)` embedding matrix is never
    materialized. Each k-means++ round only requires two matrix-vector
    products, which reduces the space complexity from
    `O(n_samples * n_classes * n_features)` to
    `O(n_samples * (n_classes + n_features))`.

    Parameters
    ----------
    clf_embedding_flag_name : dict or str or None, default=None
        Flag, which is passed to the `predict_proba` method for
        getting the (learned) sample representations.

        - If `clf_embedding_flag_name is None` and `predict_proba` returns
          only one output, the input samples `X` are used.
        - If `clf_embedding_flag_name is None` and `predict_proba` returns
          two outputs, `(proba, embeddings)` are expected as outputs.
        - If `isinstance(clf_embedding_name, str)`, we call::

            clf.predict_proba(X, **{clf_embedding_flag_name: True})

          and expect `(proba, embeddings)` as output.
        - If `isinstance(clf_embedding_name, dict)`, we call::

            clf.predict_proba(X, **clf_embedding_flag_name)

          and expect `(proba, embeddings)` as output.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal.
       Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds.
       In Int. Conf. Learn. Represent., 2020.
    .. [2] J. Zhang, Y. Chen, G. Canal, S. Mussmann, A. M. Das, G. Bhatt,
       Y. Zhu, J. Bilmes, S. S. Du, K. Jamieson, and R. D. Nowak. LabelBench:
       A Comprehensive Framework for Benchmarking Adaptive Label-Efficient
       Learning. J. Data-centric Mach. Learn. Res., 2024.
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
            - If `candidates` is of shape `(n_candidates, ...)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle. If it
            exceeds the number of unlabeled candidates, it is reduced to that
            number and a warning is raised.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. A sample is selected at most once per batch.

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
            Each row is the k-means++ sampling distribution of the respective
            round, i.e., its `nansum` is one. Utilities for labeled samples
            and for samples that have already been selected in an earlier
            round will be set to np.nan.

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

        # If `candidates` is an index array containing labeled samples, the
        # number of unlabeled candidates may fall below `batch_size`, which
        # `_validate_data` cannot detect since it only counts candidates.
        n_unlbld = len(X_unlbld)
        if n_unlbld == 0:
            raise ValueError("'candidates' contains no unlabeled samples.")
        if batch_size > n_unlbld:
            warnings.warn(
                f"'batch_size={batch_size}' is larger than number of "
                f"unlabeled candidates. Instead, "
                f"'batch_size={n_unlbld}' was set."
            )
            batch_size = n_unlbld

        # gradient embedding, aka predict class membership probabilities
        probas = clf.predict_proba(X_unlbld, **predict_proba_kwargs)
        if isinstance(probas, tuple):
            probas, X_unlbld = probas

        # Factorized gradient embedding `g_i = kron(q_i, v_i)`, where
        # `q_i = probas_i - e_{y_pred_i}` is the probability residual and
        # `v_i` the sample representation. `float64` is required because the
        # accumulation error of `float32` changes the sampling.
        probas = np.asarray(probas, dtype=np.float64)
        V = np.asarray(X_unlbld, dtype=np.float64)
        y_pred = probas.argmax(axis=-1)
        Q = probas.copy()
        Q[np.arange(n_unlbld), y_pred] -= 1
        g_norm_2 = np.einsum("ij,ij->i", Q, Q) * np.einsum("ij,ij->i", V, V)

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
        query_indices = []
        query_indices_in_unlbld = []
        # In the first round, `d_2` holds the squared gradient norms, which
        # only serve to determine the first center. Afterwards, it is replaced
        # by the squared distances to that center, such that the origin does
        # not act as a permanent ghost center in the running minimum.
        d_2 = g_norm_2.copy()
        for i in range(batch_size):
            # Zeroing the distances of the already selected centers gives them
            # zero probability, so that they cannot be drawn a second time.
            d_2[query_indices_in_unlbld] = 0
            d_2_sum = d_2.sum()
            if d_2_sum > 0:
                d_probas = d_2 / d_2_sum
            else:
                # Degenerate case of exclusively zero gradient embeddings,
                # e.g., for the one-hot probabilities of a single-class cold
                # start. Then, sample uniformly among the remaining samples.
                d_probas = np.full(n_unlbld, 1 / (n_unlbld - i))
                d_probas[query_indices_in_unlbld] = 0

            utilities[i, unlbld_mapping] = d_probas
            utilities[i, query_indices] = np.nan

            if i == 0 and d_2_sum > 0:
                idx_in_unlbld = int(np.argmax(d_2))
            else:
                idx_in_unlbld = int(
                    self.random_state_.choice(
                        n_unlbld, 1, replace=False, p=d_probas
                    )[0]
                )
            query_indices_in_unlbld.append(idx_in_unlbld)
            query_indices.append(unlbld_mapping[idx_in_unlbld])

            # Squared distance to the newest center via the factorization:
            # `||g_i - g_c||^2 = ||g_i||^2 + ||g_c||^2
            #  - 2 * <q_i, q_c> * <v_i, v_c>`. Rounding may produce tiny
            # negative values, which are rejected by `choice(p=...)`, such
            # that they are clipped.
            if i + 1 < batch_size:
                cross = (Q @ Q[idx_in_unlbld]) * (V @ V[idx_in_unlbld])
                d_2_new = g_norm_2 + g_norm_2[idx_in_unlbld] - 2 * cross
                np.maximum(d_2_new, 0, out=d_2_new)
                if i == 0:
                    d_2 = d_2_new
                else:
                    np.minimum(d_2, d_2_new, out=d_2)

        query_indices = np.array(query_indices)
        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
