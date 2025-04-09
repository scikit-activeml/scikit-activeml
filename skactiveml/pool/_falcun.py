"""
Module implementing `Falcun`, which is a deep active learning strategy jointly
selecting uncertain and diverse samples.
"""

import numpy as np

from sklearn.base import clone

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_scalar,
    check_type,
    check_equal_missing_label,
)
from ._uncertainty_sampling import uncertainty_scores


class Falcun(SingleAnnotatorPoolQueryStrategy):
    """Fast Active Learning by Contrastive UNcertainty (FALCUN)

    This class implements the "Fast Active Learning by Contrastive UNcertainty"
    (FALCUN) query strategy [1]_, which is a hybrid pool-based strategy that
    jointly selects uncertain samples via margin sampling while considering
    batch diversity within the class probability space.

    Parameters
    ----------
    gamma : float > 0, default=10
        Controls the randomness in the selection. A value of 0 corresponds to
        random sampling, while a value going to infinity corresponds to
        selecting the sample with the highest utility (relevance).
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] S. Gilhuber, A. Beer, Y. Ma, and T. Seidl. FALCUN: A Simple and
       Efficient Deep Active Learning Strategy. In Joint Eur. Conf. Mach.
       Learn. Knowl. Discov. Databases, pages 421–439, 2024.
    """

    def __init__(
        self,
        gamma=10,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.gamma = gamma

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
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`.)
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight: array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
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
        # Check parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        check_scalar(
            self.gamma,
            "gamma",
            min_val=0,
            target_type=(float, int),
            min_inclusive=True,
        )
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)

        # Fit classifier, if requested.
        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)

        # Compute uncertainties via margin sampling (cf. Eq. (1) in [1]).
        probas_cand = clf.predict_proba(X_cand)
        unc_cand = uncertainty_scores(probas_cand, method="margin_sampling")

        # Initialize distances in probability space (cf. Eq. (3) in [1]).
        dist_cand = unc_cand.copy()

        query_indices = []
        utilities_cand = np.full((batch_size, len(X_cand)), np.nan)
        cand_indices = np.arange(len(X_cand))
        for b in range(batch_size):
            if b > 0:
                # Update distances (diversity) values in the class probability
                # space (cf. Eqs. (2) and (4) in [1]).
                probas_q = probas_cand[[query_indices[int(b - 1)]]]
                dist_new = np.abs(probas_cand - probas_q).sum(axis=1)
                dist_cand = np.minimum(dist_new, dist_cand)
                dist_min = dist_cand.min()
                dist_range = dist_cand.max() - dist_min
                dist_cand -= dist_min
                if dist_range > 0:
                    dist_cand /= dist_range

            # Compute relevance scores for candidates (cf. Eq. (5) and
            # (6) in [1]).
            rel_cand = (unc_cand + dist_cand) ** self.gamma
            rel_cand[query_indices] = 0
            rel_cand_sum = np.sum(rel_cand)
            if rel_cand_sum == 0:
                rel_cand = np.ones_like(rel_cand)
                rel_cand[query_indices] = 0
            rel_cand = rel_cand / np.sum(rel_cand)

            # Sample instance to be labeled (cf. Eq. (6) in [1]).
            query_idx = self.random_state_.choice(
                cand_indices, p=rel_cand, size=1
            )
            rel_cand[query_indices] = np.nan
            utilities_cand[b] = rel_cand
            query_indices.append(query_idx[0])

        if mapping is not None:
            query_indices = mapping[query_indices]
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand
        else:
            utilities = utilities_cand

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
