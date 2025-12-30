"""
Module implementing various uncertainty based query strategies.
"""

# Authors: Pascal Mergard <Pascal.Mergard@student.uni-kassel.de>
#          Marek Herde <marek.herde@uni-kassel.de>

import numpy as np
from sklearn import clone
from sklearn.utils.validation import check_array

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_cost_matrix,
    simple_batch,
    check_classes,
    check_type,
    check_equal_missing_label,
)


class UncertaintySampling(SingleAnnotatorPoolQueryStrategy):
    """Uncertainty Sampling (US)

    This class implement various uncertainty based query strategies, i.e., the
    standard uncertainty measures [1]_, cost-sensitive ones [2]_, and one
    optimizing expected average precision [3]_. Concretely, four uncertainty
    score definitions are available:

    - least confident [1]_ selecting samples whose predicted top
      class has the lowest confidence,
    - margin sampling [1]_ selecting samples where the gap between the two most
      probable classes is smallest,
    - entropy-based uncertainty [1]_ selecting samples with the highest overall
      predictive uncertainty across classes,
    - and expected average precision [3] selecting samples with the highest
      expected improvement in average precision under the model's current
      predictions.

    For the least confident and margin sampling cost-sensitive variants
    are implemented considering a user-defined cost matrix [2]_. Finally,
    the class can also be leverage to implement variants of density-weighted
    uncertainty sampling (DWUS) and the dual strategy for active learning
    (DUAL) [4]_.

    Parameters
    ----------
    method : 'least_confident' or 'margin' or 'entropy' or \
            'expected_average_precision', default='least_confident'
        The method to calculate the uncertainty.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`. Only supported for
        `least_confident` and `margin_sampling` variant.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState
        The random state to use.

    References
    ----------
    .. [1] Settles, Burr. Active learning literature survey. University of
       Wisconsin-Madison Department of Computer Sciences, 2009.

    .. [2] P.-L. Chen and H.-T. Lin. Active Learning for Multiclass
       Cost-Sensitive Classification Using Probabilistic Models. In Conf.
       Technol. Appl. Artif. Intell., pages 13–18, 2013.

    .. [3] H. Wang, X. Chang, L. Shi, Y. Yang, and Y.-D. Shen. Uncertainty
       Sampling for Action Recognition via Maximizing Expected Average
       Precision. In Int. Jt. Conf. Artif. Intell., pages 964–970, 2018.

    .. [4] P. Donmez, J. G. Carbonell, and P. N. Bennett. Dual Strategy Active
       Learning. In Eur. Conf. Mach. Learn, pages 116-127, 2007.

    """

    def __init__(
        self,
        method="least_confident",
        cost_matrix=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.method = method
        self.cost_matrix = cost_matrix

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        utility_weight=None,
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
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        utility_weight : array-like, default=None
            Weight for each candidate (multiplied with utilities). Usually,
            this is to be the density of a candidate. The length of
            `utility_weight` is usually `n_samples`, except for the case when
            `candidates` contains samples (ndim >= 2). Then the length is
            `n_candidates`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
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
            - If `candidates` is of shape `(n_candidates, ...)`, `utilities`
              refers to the indexing in `candidates`.
        """
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate classifier type.
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        # Validate classifier type.
        check_type(fit_clf, "fit_clf", bool)

        # Check `utility_weight`.
        if utility_weight is None:
            if mapping is None:
                utility_weight = np.ones(len(X_cand))
            else:
                utility_weight = np.ones(len(X))
        utility_weight = check_array(utility_weight, ensure_2d=False)

        if mapping is None and not len(X_cand) == len(utility_weight):
            raise ValueError(
                f"'utility_weight' must have length 'n_candidates' but "
                f"{len(X_cand)} != {len(utility_weight)}."
            )
        if mapping is not None and not len(X) == len(utility_weight):
            raise ValueError(
                f"'utility_weight' must have length 'n_samples' but "
                f"{len(utility_weight)} != {len(X)}."
            )

        # Validate method.
        if not isinstance(self.method, str):
            raise TypeError(
                "{} is an invalid type for method. Type {} is "
                "expected".format(type(self.method), str)
            )

        # sample_weight is checked by clf when fitted

        # Fit the classifier.
        if fit_clf:
            if sample_weight is not None:
                clf = clone(clf).fit(X, y, sample_weight)
            else:
                clf = clone(clf).fit(X, y)

        # Predict class-membership probabilities.
        probas = clf.predict_proba(X_cand)

        # Choose the method and calculate corresponding utilities.
        with np.errstate(divide="ignore"):
            if self.method in [
                "least_confident",
                "margin_sampling",
                "entropy",
            ]:
                utilities_cand = uncertainty_scores(
                    probas=probas,
                    method=self.method,
                    cost_matrix=self.cost_matrix,
                )
            elif self.method == "expected_average_precision":
                classes = clf.classes_
                utilities_cand = expected_average_precision(classes, probas)
            else:
                raise ValueError(
                    "The given method {} is not valid. Supported methods are "
                    "'entropy', 'least_confident', 'margin_sampling' and "
                    "'expected_average_precision'".format(self.method)
                )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand
        utilities *= utility_weight

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def uncertainty_scores(probas, cost_matrix=None, method="least_confident"):
    """Computes uncertainty scores. Three methods are available: least
    confident ('least_confident'), margin sampling ('margin_sampling'),
    and entropy based uncertainty ('entropy') [1]_. For the least confident and
    margin sampling methods cost-sensitive variants are implemented in case of
    a given cost matrix (see [2]_ for more information).

    Parameters
    ----------
    probas : array-like of shape (n_samples, n_classes)
        Class membership probabilities for each sample.
    cost_matrix : array-like pf shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`. Only supported for
        'least_confident' or 'margin_sampling'.
    method : 'least_confident' or 'margin_sampling' or 'entropy', \
            default='least_confident'
        The method to calculate the uncertainty.

    References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey".
       University of Wisconsin-Madison Department of Computer Sciences, 2009.
    .. [2] P.-L. Chen and H.-T. Lin. Active Learning for Multiclass
       Cost-Sensitive Classification Using Probabilistic Models. In Conf.
       Technol. Appl. Artif. Intell., pages 13–18, 2013.
    """
    # Check probabilities.
    probas = check_array(probas)

    if not np.allclose(np.sum(probas, axis=1), 1, rtol=0, atol=1.0e-3):
        raise ValueError(
            "'probas' are invalid. The sum over axis 1 must be one."
        )

    n_classes = probas.shape[1]

    # Check cost matrix.
    if cost_matrix is not None:
        cost_matrix = check_cost_matrix(cost_matrix, n_classes=n_classes)

    # Compute uncertainties.
    if method == "least_confident":
        if cost_matrix is None:
            return 1 - np.max(probas, axis=1)
        else:
            costs = probas @ cost_matrix
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return costs[:, 0]
    elif method == "margin_sampling":
        if cost_matrix is None:
            probas = -(np.partition(-probas, 1, axis=1)[:, :2])
            return 1 - np.abs(probas[:, 0] - probas[:, 1])
        else:
            costs = probas @ cost_matrix
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return -np.abs(costs[:, 0] - costs[:, 1])
    elif method == "entropy":
        if cost_matrix is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.nansum(-probas * np.log(probas), axis=1)
        else:
            raise ValueError(
                "Method `entropy` does not support cost matrices but "
                "`cost_matrix` was not None."
            )
    else:
        raise ValueError(
            "Supported methods are ['least_confident', 'margin_sampling', "
            "'entropy'], the given one is: {}.".format(method)
        )


def expected_average_precision(classes, probas):
    """
    Calculate the expected average precision [1]_.

    Parameters
    ----------
    classes : array-like, shape=(n_classes,)
        Holds the label for each class.
    probas : array-like of shape (n_samples, n_classes)
        Class membership probabilities for each sample.

    Returns
    -------
    score : np.ndarray of shape=(n_samples,)
        The expected average precision score of all samples in candidates.

    References
    ----------
    .. [1] H. Wang, X. Chang, L. Shi, Y. Yang, and Y.-D. Shen. Uncertainty
       Sampling for Action Recognition via Maximizing Expected Average
       Precision. In Int. Jt. Conf. Artif. Intell., pages 964–970, 2018.
    """
    # Check if `probas` is valid.
    probas = check_array(
        probas,
        accept_sparse=False,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        ensure_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        estimator=None,
    )

    if (np.sum(probas, axis=1) - 1).all():
        raise ValueError(
            "probas are invalid. The sum over axis 1 must be " "one."
        )

    # Check if `classes` are valid.
    check_classes(classes)
    if len(classes) < 2:
        raise ValueError("`classes` must contain at least 2 entries.")
    if len(classes) != probas.shape[1]:
        raise ValueError(
            "`classes` must have the same length as `probas` has " "columns."
        )

    score = np.zeros(len(probas))
    probas_sort_idx_per_class = np.argsort(-probas, axis=0)
    probas_sorted = np.take_along_axis(
        probas, probas_sort_idx_per_class, axis=0
    )
    
    g_arr_mask = np.arange(probas.shape[0] - 1) > 0
    f_arr_mask = np.arange(probas.shape[0]) > 0
    for i in range(len(classes)):
        for j in range(len(probas)):
            # The i-th column of p without p[j,i]
            p = np.delete(probas_sorted[:, i], [j])
            # calculate g_arr
            g_arr = np.zeros((len(p), len(p)))
            if len(p) > 0:
                g_arr[0, 0] = 1
            for n in range(1, len(p)):
                # p_n*g(n-1,t-1)
                g_term_0 = p[n - 1] * np.pad(g_arr[n - 1], (1, 0))[:-1]
                # (1-p_n)*g(n-1,t)
                g_term_1 = (1 - p[n - 1]) * g_arr[n - 1]
                g_arr[n] = (g_term_0 + g_term_1) * (g_arr_mask)

            # calculate f_arr
            f_arr = np.zeros((len(p) + 1, len(p) + 1))
            f_arr[0, 0] = 1
            for n in range(1, len(p) + 1):
                # p_n*f(n-1,t-1)
                f_term_0 = p[n - 1] * np.pad(f_arr[n - 1], (1, 0))[:-1]
                # p_n*t/n*g(n-1,t-1)
                f_term_1 = p[n - 1] * np.pad(g_arr[n - 1], (1, 0))
                f_term_1 *= np.arange(len(p) + 1) / n
                # (1-p_n)*f(n-1,t)
                f_term_2 = (1 - p[n - 1]) * f_arr[n - 1]
                f_arr[n] = (f_term_0 + f_term_1 + f_term_2) * f_arr_mask

            # calculate score
            sample_index = probas_sort_idx_per_class[j, i]
            score[sample_index] += np.sum(
                f_arr[len(p), 1:] / np.arange(1, len(p) + 1)
            )

    return score
