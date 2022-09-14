"""
Query-by-committee strategies.
"""

# Author: Pascal Mergard <Pascal.Mergard@student.uni-kassel.de>
#         Marek Herde <marek.herde@uni-kassel.de>
import copy

import numpy as np
from sklearn import clone
from sklearn.utils.validation import check_array, _is_arraylike

from ..base import (
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
    SkactivemlRegressor,
)
from ..utils import (
    simple_batch,
    check_type,
    compute_vote_vectors,
    MISSING_LABEL,
    check_equal_missing_label,
)


class QueryByCommittee(SingleAnnotatorPoolQueryStrategy):
    """Query-by-Committee.

    The Query-by-Committee (QueryByCommittee) strategy uses an ensemble of
    estimators to identify on which instances many estimators disagree.

    Parameters
    ----------
    method : string, default='KL_divergence'
        The method to calculate the disagreement in the case of classification.
        KL_divergence or vote_entropy are possible. In the case of regression
        the empirical variance is used.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.
    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and
        bagging. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 1-9. Morgan Kaufmann, 1998.
    [3] Burbidge, Robert and Rowland, Jem J and King, Ross D. Active learning
        for regression based on query by committee. International conference on
        intelligent data engineering and automated learning, pages 209--218,
        2007.
    """

    def __init__(
        self,
        method="KL_divergence",
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.method = method

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
        ensemble : list or tuple of SkactivemlClassifier or list or tuple of
        SkactivemlRegressor, SkactivemlClassifier or SkactivemlRegressor
            If `ensemble` is a `SkactivemlClassifier` or a `SkactivemlRegressor`
            , it must have `n_estimators` and `estimators_` after fitting as
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

        ensemble, est_arr, classes = _check_ensemble(
            ensemble=ensemble,
            X=X,
            y=y,
            sample_weight=sample_weight,
            fit_ensemble=fit_ensemble,
            missing_label=self.missing_label_,
            estimator_types=[SkactivemlClassifier, SkactivemlRegressor],
        )
        # classes is None if the ensemble is a regressor
        if classes is not None:
            check_type(
                self.method,
                "method",
                target_vals=["KL_divergence", "vote_entropy"],
            )

            # Compute utilities.
            if self.method == "KL_divergence":
                probas = np.array(
                    [est.predict_proba(X_cand) for est in est_arr]
                )
                utilities_cand = average_kl_divergence(probas)
            else:  # self.method == "vote_entropy":
                votes = np.array([est.predict(X_cand) for est in est_arr]).T
                utilities_cand = vote_entropy(votes, classes)

        else:
            results = np.array(
                [learner.predict(X_cand) for learner in est_arr]
            )
            utilities_cand = np.std(results, axis=0)

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def average_kl_divergence(probas):
    """Calculates the average Kullback-Leibler (KL) divergence for measuring
    the level of disagreement in QueryByCommittee.

    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.

    Returns
    -------
    scores: np.ndarray, shape (n_samples)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning
        for text classification. In Proceedings of the International Conference
        on Machine Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """
    # Check probabilities.
    probas = check_array(probas, allow_nd=True)
    if probas.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got {probas.ndim}D array instead."
        )
    n_estimators = probas.shape[0]

    # Calculate the average KL divergence.
    probas_mean = np.mean(probas, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.nansum(
            np.nansum(probas * np.log(probas / probas_mean), axis=2), axis=0
        )
    scores = scores / n_estimators

    return scores


def vote_entropy(votes, classes):
    """Calculates the vote entropy for measuring the level of disagreement in
    QueryByCommittee.

    Parameters
    ----------
    votes : array-like, shape (n_samples, n_estimators)
        The class predicted by the estimators for each sample.
    classes : array-like, shape (n_classes)
        A list of all possible classes.

    Returns
    -------
    vote_entropy : np.ndarray, shape (n_samples)
        The vote entropy of each row in `votes`.

    References
    ----------
    [1] Engelson, Sean P., and Ido Dagan.
        Minimizing manual annotation cost in supervised training from corpora.
        arXiv preprint cmp-lg/9606030 (1996).
    """
    # Check `votes` array.
    votes = check_array(votes)
    n_estimators = votes.shape[1]

    # Count the votes.
    vote_count = compute_vote_vectors(
        y=votes, classes=classes, missing_label=None
    )

    # Compute vote entropy.
    v = vote_count / n_estimators

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.nansum(-v * np.log(v), axis=1)
    return scores


def _check_ensemble(
    ensemble,
    estimator_types,
    X,
    y,
    sample_weight,
    fit_ensemble=True,
    missing_label=MISSING_LABEL,
):
    # Check if the parameter `ensemble` is valid.
    for estimator_type in estimator_types:
        if isinstance(ensemble, estimator_type) and (
            hasattr(ensemble, "n_estimators")
            or hasattr(ensemble, "estimators")
        ):
            check_equal_missing_label(ensemble.missing_label, missing_label)
            # Fit the ensemble.
            if fit_ensemble:
                ensemble = clone(ensemble).fit(X, y, sample_weight)

            if hasattr(ensemble, "estimators_"):
                est_arr = ensemble.estimators_
            else:
                if hasattr(ensemble, "estimators"):
                    n_estimators = len(ensemble.estimators)
                else:
                    n_estimators = ensemble.n_estimators
                est_arr = [ensemble] * n_estimators

            if estimator_type == SkactivemlClassifier:
                return ensemble, est_arr, ensemble.classes_
            else:
                return ensemble, est_arr, None

        elif isinstance(ensemble, (list, tuple)) and isinstance(
            ensemble[0], estimator_type
        ):
            est_arr = copy.deepcopy(ensemble)
            for i in range(len(est_arr)):
                check_type(
                    est_arr[i], f"ensemble[{i}]", estimator_type
                )  # better error message
                check_equal_missing_label(
                    est_arr[i].missing_label, missing_label
                )
                # Fit the ensemble.
                if fit_ensemble:
                    est_arr[i] = est_arr[i].fit(X, y, sample_weight)

                if i > 0 and estimator_type == SkactivemlClassifier:
                    np.testing.assert_array_equal(
                        est_arr[i - 1].classes_,
                        est_arr[i].classes_,
                        err_msg=f"The inferred classes of the {i - 1}-th and "
                        f"{i}-th are not equal. Set the `classes` "
                        f"parameter of each ensemble member to avoid "
                        f"this error.",
                    )
            if estimator_type == SkactivemlClassifier:
                return ensemble, est_arr, est_arr[0].classes_
            else:
                return ensemble, est_arr, None

    raise TypeError(
        f"`ensemble` must either be a `{estimator_types} "
        f"with the attribute `n_ensembles` and `estimators_` after "
        f"fitting or a list of {estimator_types} objects."
    )
