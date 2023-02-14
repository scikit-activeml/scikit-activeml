import numpy as np
from setuptools._distutils.command.check import check
from sklearn.utils import check_array

from ..base import SkactivemlClassifier, SkactivemlRegressor
from ..pool._query_by_committee import _check_ensemble, QueryByCommittee
from ..utils import rand_argmax, MISSING_LABEL, check_type, check_scalar, \
    check_random_state


class BALD(QueryByCommittee):
    """Batch Bayesian Active Learning by Disagreement (BatchBALD)

    The Bayesian-Active-Learning-by-Disagreement (BALD) [1] strategy reduces the
    number  of possible hypotheses maximally fast to minimize the uncertainty
    about the parameters using Shannon’s entropy. It seeks the data point that
    maximises the decrease in expected posterior entropy. For the batch case
    the advanced strategy BatchBALD [2] is applied.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    [1] Houlsby, Neil, et al. Bayesian active learning for classification and
        preference learning. arXiv preprint arXiv:1112.5745, 2011.
    [2] Kirsch, Andreas; Van Amersfoort, Joost; GAL, Yarin.
        Batchbald: Efficient and diverse batch acquisition for deep bayesian
        active learning. Advances in neural information processing systems,
        2019, 32. Jg.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self,
        X,
        y,
        ensemble,
        fit_ensemble=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.)
        ensemble : list or tuple of SkactivemlClassifier or
            SkactivemlClassifier.
            If `ensemble` is a `SkactivemlClassifier`, it must have
            `n_estimators` and `estimators_` after fitting as
            attribute. Then, its estimators will be used as committee. If
            `ensemble` is array-like, each element of this list must be
            `SkactivemlClassifier` or a `SkactivemlRegressor` and will be used
            as committee member.
        fit_ensemble : bool, default=True
            Defines whether the ensemble should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
                array-like of shape (n_candidates, n_features), default=None
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

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
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate classifier type.
        check_type(fit_ensemble, "fit_ensemble", bool)

        ensemble, est_arr, _ = _check_ensemble(
            ensemble=ensemble,
            X=X,
            y=y,
            sample_weight=sample_weight,
            fit_ensemble=fit_ensemble,
            missing_label=self.missing_label_,
            estimator_types=[SkactivemlClassifier],
        )

        probas = np.array(
                [est.predict_proba(X_cand) for est in est_arr]
            )
        batch_utilities_cand = batch_bald(probas, batch_size,
                                          self.random_state_)

        if mapping is None:
            batch_utilities = batch_utilities_cand
        else:
            batch_utilities = np.full((batch_size, len(X)), np.nan)
            batch_utilities[:, mapping] = batch_utilities_cand

        best_indices = rand_argmax(batch_utilities,
                                   axis=1,
                                   random_state=self.random_state_)

        if return_utilities:
            return best_indices, batch_utilities
        else:
            return best_indices


def batch_bald(probas, batch_size=1, random_state=None):
    """BatchBALD: Efficient and Diverse Batch Acquisition
        for Deep Bayesian Active Learning

    BatchBALD [1] is an extension of BALD (Bayesian Active Learning by
    Disagreement) [2] whereby points are jointly scored by estimating the
    mutual information between a joint of multiple data points and the model
    parameters.

    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.
    batch_size : int, default=1
        The number of samples to be selected in one AL cycle.
    random_state : int or np.random.RandomState, default=None
        The random state to use.

    Returns
    -------
    scores: np.ndarray, shape (n_samples)
        The BatchBALD-scores.

    References
    ----------
    [1] Kirsch, Andreas, Joost Van Amersfoort, and Yarin Gal. "Batchbald:
        Efficient and diverse batch acquisition for deep bayesian active
        learning." Advances in neural information processing systems 32 (2019).
    [2] Houlsby, Neil, et al. "Bayesian active learning for classification and
        preference learning." arXiv preprint arXiv:1112.5745 (2011).
    """
    # Validate input parameters.
    if probas.ndim != 3:
        raise ValueError(f"'probas' should be of shape 3, but {probas.ndim}"
                         f" were given.")
    probas = check_array(probas, ensure_2d=False, allow_nd=True,
                         force_all_finite="allow-nan")
    check_scalar(batch_size, "batch_size", int, min_val=1)
    check_random_state(random_state)

    n_estimators, n_samples, n_classes = probas.shape
    utils = np.full((batch_size, probas.shape[1]), fill_value=np.nan)
    batch = np.empty(0, dtype=np.int64)
    # Eq. 12 in paper:
    confidents = np.nanmean(np.nansum(-probas * np.log(probas), axis=2),
                            axis=0)
    confident = 0
    P = np.ones((1, n_estimators))
    for n in range(batch_size):
        # Eq. 13 in paper:
        P_ = (1 / n_estimators) * P @ np.swapaxes(probas, 0, 1)
        # Eq. 12 in paper:
        scores = -np.sum(P_ * np.log(P_), axis=(1, 2))
        # Eq. 9 in paper:
        scores -= confident + confidents

        scores[batch] = np.nan
        idx = rand_argmax(scores, random_state=random_state)
        if n == 0:
            P = probas[:, idx[0]].T
        else:
            P = np.append(P, probas[:, idx[0]].T, axis=0)
        confident += confidents[idx]
        batch = np.append(batch, idx)
        utils[n] = scores
    return utils
