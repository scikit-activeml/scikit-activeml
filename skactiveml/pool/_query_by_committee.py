"""
Query-by-committee strategies.
"""

# Author: Pascal Mergard <Pascal.Mergard@student.uni-kassel.de>
#         Marek Herde <marek.herde@uni-kassel.de>
import copy

import numpy as np
from sklearn import clone
from sklearn.utils.validation import check_array, check_is_fitted
from iteration_utilities import flatten

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
    check_scalar,
)


class QueryByCommittee(SingleAnnotatorPoolQueryStrategy):
    """Query-by-Committee (QBC)

    The Query-by-Committee (QBC) [1]_, [2]_, [3]_, [4]_, [5]_ strategy uses an
    ensemble of estimators to identify on which samples many estimators
    disagree.

    Parameters
    ----------
    method : "KL_divergence" or "vote_entropy" or "variation_ratios, \
            default='KL_divergence'
        The method to calculate the disagreement in the case of classification.
        'KL_divergence', 'vote_entropy', and 'variation_ratios' are possible.
        In the case of regression, this parameter is ignored and the empirical
        variance is used.
    eps : float > 0, default=1e-7
        Minimum probability threshold to compute log-probabilities (only
        relevant for `method='KL_divergence'`).
    sample_predictions_method_name : str, default=None
        Certain estimators may offer methods enabling to construct a committee
        by sampling predictions of committee members. This parameter is to
        indicate the name of such a method.

        - If `sample_predictions_method_name=None` no sampling is
          performed.
        - If `sample_predictions_method_name` is not `None` and in the
          case of classification, the method is expected to take samples of
          the shape `(n_samples, *)` as input and to output probabilities
          of the shape `(n_members, n_samples, n_classes)`, e.g.,
          `sample_proba` in `skactiveml.base.ClassFrequencyEstimator`.
        - If `sample_predictions_method_name` is not `None` and in the
          case of regression, the method is expected to take samples of the
          shape `(n_samples, *)` as input and to output numerical values of
          the shape `(n_members, n_samples)`, e.g., `sample_y` in
          `sklearn.gaussian_process.GaussianProcessRegressor`.
    sample_predictions_dict : dict, default=None
        Parameters (excluding the samples) that are passed to the method with
        the name `sample_predictions_method_name`.

        - This parameter must be `None`, if
          `sample_predictions_method_name` is `None`.
        - Otherwise, it may be used to define the number of sampled
          members, e.g., by defining `n_samples` as parameter to the method
          `sample_proba` of `skactiveml.base.ClassFrequencyEstimator` or
          `sample_y` of
          `sklearn.gaussian_process.GaussianProcessRegressor`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState or None, default=None
        The random state to use.

    References
    ----------
    .. [1] H. S. Seung, M. Opper, and H. Sompolinsky. Query by Committee. In
       Annu. Workshop Comput. Learn. Theory., pages 287–294, 1992.

    .. [2] A. K. McCallum and K. Nigamy. Employing EM and Pool-Based Active
       Learning for Text Classification. In Int. Conf. Mach. Learn.,
       pages 359–367, 1998.

    .. [3] S. P. Engelson and I. Dagan. Minimizing Manual Annotation Cost in
       Supervised Training from Corpora. In Annu. Meet. Assoc. Comput.
       Linguist., pages 319–326, 1996.

    .. [4] R. Burbidge, J. J. Rowland, and R. D. King. Active Learning for
       Regression Based on Query by Committee. In Intell. Data Eng. Autom.
       Learn., pages 209–218, 2007.

    .. [5] W. H. Beluch, T. Genewein, A. Nürnberger, and J. M. Köhler. The
       Power of Ensembles for Active Learning in Image Classification. In
       IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 9368–9377, 2018.
    """

    def __init__(
        self,
        method="KL_divergence",
        eps=1e-7,
        sample_predictions_method_name=None,
        sample_predictions_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.method = method
        self.eps = eps
        self.sample_predictions_method_name = sample_predictions_method_name
        self.sample_predictions_dict = sample_predictions_dict

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
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`.)
        ensemble : array-like of SkactivemlClassifier or array-like of \
                SkactivemlRegressor or SkactivemlClassifier or \
                SkactivemlRegressor
            - If `ensemble` is a `SkactivemlClassifier` or a
              `SkactivemlRegressor` and has `n_estimators` plus
              `estimators_` after fitting as attributes, its estimators will
              be used as committee.
            - If `ensemble` is array-like, each element of this list must be
              `SkactivemlClassifier` or a `SkactivemlRegressor` and will be
              used as committee member.
            - If `ensemble` is a `SkactivemlClassifier` or a
              `SkactivemlRegressor` and implements a method with the name
              `sample_predictions_method_name`, this method is used to sample
              predictions of committee members.
        fit_ensemble : bool, default=True
            Defines whether the ensemble should be fitted on `X`, `y`, and
            `sample_weight`.
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
              necessarily contained in `X`). This is not supported by all
              query strategies.
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
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)
        check_type(fit_ensemble, "fit_ensemble", bool)
        ensemble, est_arr, classes, sample_func, sample_dict = _check_ensemble(
            ensemble=ensemble,
            X=X,
            y=y,
            sample_weight=sample_weight,
            fit_ensemble=fit_ensemble,
            missing_label=self.missing_label_,
            estimator_types=[SkactivemlClassifier, SkactivemlRegressor],
            sample_predictions_method_name=self.sample_predictions_method_name,
            sample_predictions_dict=self.sample_predictions_dict,
        )
        check_type(
            self.method,
            "method",
            target_vals=["KL_divergence", "vote_entropy", "variation_ratios"],
        )

        # `classes` is `None`, if `ensemble` is a regressor.
        if classes is not None:
            # Compute utilities.
            if self.method == "KL_divergence":
                if sample_func is None:
                    probas = self._aggregate_predict_probas(
                        X_cand, ensemble, est_arr
                    )
                else:
                    probas = sample_func(X_cand, **sample_dict)
                utilities_cand = average_kl_divergence(probas, self.eps)
            else:
                if sample_func is None:
                    votes = np.array(
                        [est.predict(X_cand) for est in est_arr]
                    ).T
                else:
                    probas = sample_func(X_cand, **sample_dict)
                    votes = probas.argmax(axis=-1).T
                if self.method == "vote_entropy":
                    utilities_cand = vote_entropy(votes, classes)
                else:
                    utilities_cand = variation_ratios(votes)
        else:
            if sample_func is None:
                results = np.array(
                    [learner.predict(X_cand) for learner in est_arr]
                )
            else:
                results = sample_func(X_cand, **sample_dict).T
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

    def _aggregate_predict_probas(self, X_cand, ensemble, est_arr):
        """Aggregate the predicted probabilities across all ensemble members
        and ensure that all classes are mapped correctly.

        Parameters
        ----------
        X_cand : array-like of shape (n_samples, n_features)
            Samples whose probabilities are to be predicted.
        ensemble : SkactivemlClassifier or list or tuple of \
                SkactivemlClassifier
            - If `ensemble` is a `SkactivemlClassifier`, it must have
              `n_estimators` and `estimators_` after fitting as attribute.
              Then, its estimators will be used as committee.
            - If `ensemble` is array-like, each element of this list must be
              `SkactivemlClassifier` and will be used as committee member.
        est_arr : list or tuple of SkactivemlClassifier
            List of ensemble members contained in `ensemble`.

        Returns
        -------
        probas : np.ndarray of shape (n_samples, n_classes)
            The mapped predicted probabilities.
        """
        if hasattr(ensemble, "classes_"):
            ensemble_classes = ensemble.classes_
        else:
            ensemble_classes = np.unique(
                list(flatten([est.classes_ for est in est_arr]))
            )
        probas = np.zeros((len(est_arr), len(X_cand), len(ensemble_classes)))
        for i, est in enumerate(est_arr):
            est_proba = est.predict_proba(X_cand)
            est_classes = est.classes_

            if len(est_classes) == len(ensemble_classes):
                indices_ensemble = np.arange(len(ensemble_classes))
            else:
                indices_est = np.where(np.isin(est_classes, ensemble_classes))[
                    0
                ]
                indices_ensemble = np.searchsorted(
                    ensemble_classes, est_classes[indices_est]
                )
            probas[i, :, indices_ensemble] = est_proba.T
        return probas


def average_kl_divergence(probas, eps=1e-7):
    """Calculates the average Kullback-Leibler (KL) divergence for measuring
    the level of disagreement in QueryByCommittee.

    Parameters
    ----------
    probas : array-like of shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.
    eps : float  > 0, optional (default=1e-7)
        Minimum probability threshold to compute log-probabilities.

    Returns
    -------
    scores : np.ndarray, shape (n_samples,)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    .. [1] A. K. McCallum and K. Nigamy. Employing EM and Pool-Based Active
       Learning for Text Classification. In Int. Conf. Mach. Learn.,
       pages 359–367, 1998.
    """
    # Check parameters.
    check_scalar(
        eps,
        "eps",
        min_val=0,
        max_val=0.1,
        target_type=(float, int),
        min_inclusive=False,
    )
    probas = check_array(probas, allow_nd=True)
    if probas.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got {probas.ndim}D array instead."
        )
    n_estimators = probas.shape[0]

    np.clip(probas, a_min=eps, a_max=1, out=probas)
    probas /= probas.sum(axis=2, keepdims=True)

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
    vote_entropy : np.ndarray of shape (n_samples,)
        The vote entropy of each row in `votes`.

    References
    ----------
    .. [1] S. P. Engelson and I. Dagan. Minimizing Manual Annotation Cost in
       Supervised Training from Corpora. In Annu. Meet. Assoc. Comput.
       Linguist., pages 319–326, 1996.
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


def variation_ratios(votes):
    """Calculates the variation ratios for measuring the level of disagreement
    in `QueryByCommittee`.

    Parameters
    ----------
    votes : array-like of shape (n_samples, n_estimators)
        The class predicted by the estimators for each sample.

    Returns
    -------
    scores : np.ndarray of shape (n_samples,)
        The variation ratios of each row in `votes`.

    References
    ----------
    .. [1] W. H. Beluch, T. Genewein, A. Nürnberger, and J. M. Köhler. The
       Power of Ensembles for Active Learning in Image Classification. In
       IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 9368–9377, 2018.
    """
    # Check `votes` array.
    votes = check_array(votes)
    n_estimators = votes.shape[1]

    # Count the votes.
    vote_count = compute_vote_vectors(y=votes, missing_label=None)
    scores = 1 - (vote_count.max(axis=-1) / n_estimators)

    return scores


def _check_ensemble(
    ensemble,
    estimator_types,
    X,
    y,
    sample_weight,
    fit_ensemble=True,
    missing_label=MISSING_LABEL,
    sample_predictions_method_name=None,
    sample_predictions_dict=None,
):
    error_msg = (
        f"`ensemble` must either be a `{estimator_types} "
        f"with the attribute `n_ensembles` or `estimators_` or an array-like "
        f"of {estimator_types} objects or implement a method with the name "
        f"`{sample_predictions_method_name}`."
    )

    # Check if the parameter `ensemble` is valid.
    for estimator_type in estimator_types:
        if isinstance(ensemble, estimator_type):
            check_equal_missing_label(ensemble.missing_label, missing_label)
            # Fit the ensemble.
            if fit_ensemble:
                if sample_weight is None:
                    ensemble = clone(ensemble).fit(X, y)
                else:
                    ensemble = clone(ensemble).fit(X, y, sample_weight)
            else:
                check_is_fitted(ensemble)

            if sample_predictions_method_name is not None:
                check_type(
                    sample_predictions_method_name,
                    "sample_predictions_method_name",
                    str,
                )
                if not hasattr(ensemble, sample_predictions_method_name):
                    raise ValueError(
                        "If `sample_predictions_method_name` is not `None`, "
                        "`ensemble` must implement a method with this name."
                    )
                sample_func = getattr(ensemble, sample_predictions_method_name)
                if sample_predictions_dict is None:
                    sample_predictions_dict = {}
                if not isinstance(sample_predictions_dict, dict):
                    raise ValueError(
                        "`sample_predictions_dict` must be a `dict`, if "
                        "`sample_predictions_method_name` is not `None`."
                    )
            else:
                sample_func = None
                if sample_predictions_dict is not None:
                    raise ValueError(
                        "`sample_predictions_dict` must be `None`, if "
                        "`sample_predictions_method_name` is `None`."
                    )

            if sample_func is not None:
                est_arr = None
            elif hasattr(ensemble, "estimators_"):
                est_arr = ensemble.estimators_
            elif hasattr(ensemble, "estimators"):
                est_arr = [ensemble] * len(ensemble.estimators)
            elif hasattr(ensemble, "n_estimators"):
                est_arr = [ensemble] * ensemble.n_estimators
            else:
                raise TypeError(error_msg)

            cls = getattr(ensemble, "classes_", None)
            return ensemble, est_arr, cls, sample_func, sample_predictions_dict

        elif isinstance(ensemble, (list, tuple)) and isinstance(
            ensemble[0], estimator_type
        ):
            if (
                sample_predictions_dict is not None
                or sample_predictions_method_name is not None
            ):
                raise ValueError(
                    "`sample_predictions_method_name` and "
                    "`sample_predictions_dict` must be `None`, if `ensemble` "
                    "is array-like."
                )
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
                    if sample_weight is None:
                        est_arr[i] = est_arr[i].fit(X, y)
                    else:
                        est_arr[i] = est_arr[i].fit(X, y, sample_weight)
                else:
                    check_is_fitted(est_arr[i])

                if i > 0 and estimator_type == SkactivemlClassifier:
                    np.testing.assert_array_equal(
                        est_arr[i - 1].classes_,
                        est_arr[i].classes_,
                        err_msg=f"The inferred classes of the {i - 1}-th and "
                        f"{i}-th are not equal. Set the `classes` "
                        f"parameter of each ensemble member to avoid "
                        f"this error.",
                    )
            cls = getattr(est_arr[0], "classes_", None)
            return ensemble, est_arr, cls, None, None

    raise TypeError(error_msg)
