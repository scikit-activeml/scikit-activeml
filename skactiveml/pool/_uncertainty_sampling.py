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
    """Uncertainty Sampling.

    This class implement various uncertainty based query strategies, i.e., the
    standard uncertainty measures [1], cost-sensitive ones [2], and one
    optimizing expected average precision [3].

    Parameters
    ----------
    method : string, default='least_confident'
        The method to calculate the uncertainty, entropy, least_confident,
        margin_sampling, and expected_average_precision  are possible.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with cost_matrix[i,j] defining the cost of predicting class
        j for a sample with the actual class i. Only supported for
        `least_confident` and `margin_sampling` variant.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] Settles, Burr. Active learning literature survey.
        University of Wisconsin-Madison Department of Computer Sciences, 2009.
    [2] Chen, Po-Lung, and Hsuan-Tien Lin. "Active learning for multiclass
        cost-sensitive classification using probabilistic models." 2013
        Conference on Technologies and Applications of Artificial Intelligence.
        IEEE, 2013.
    [3] Wang, Hanmo, et al. "Uncertainty sampling for action recognition
        via maximizing expected average precision."
        IJCAI International Joint Conference on Artificial Intelligence. 2018.
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
            indicated by self.MISSING_LABEL.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
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
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        # Validate classifier type.
        check_type(fit_clf, "fit_clf", bool)

        # Validate method.
        if not isinstance(self.method, str):
            raise TypeError(
                "{} is an invalid type for method. Type {} is "
                "expected".format(type(self.method), str)
            )

        # sample_weight is checked by clf when fitted

        # Fit the classifier.
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)

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

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def uncertainty_scores(probas, cost_matrix=None, method="least_confident"):
    """Computes uncertainty scores. Three methods are available: least
    confident ('least_confident'), margin sampling ('margin_sampling'),
    and entropy based uncertainty ('entropy') [1]. For the least confident and
    margin sampling methods cost-sensitive variants are implemented in case of
    a given cost matrix (see [2] for more information).

    Parameters
    ----------
    probas : array-like, shape (n_samples, n_classes)
        Class membership probabilities for each sample.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i. Only supported for least confident
        variant.
    method : {'least_confident', 'margin_sampling', 'entropy'},
            optional (default='least_confident')
        Least confidence (lc) queries the sample whose maximal posterior
        probability is minimal. In case of a given cost matrix, the maximial
        expected cost variant is used. Smallest margin (sm) queries the sample
        whose posterior probability gap between the most and the second most
        probable class label is minimal. In case of a given cost matrix, the
        cost-weighted minimum margin is used. Entropy ('entropy') queries the
        sample whose posterior's have the maximal entropy. There is no
        cost-sensitive variant of entropy based uncertainty sampling.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey".
        University of Wisconsin-Madison Department of Computer Sciences, 2009.
    [2] Chen, Po-Lung, and Hsuan-Tien Lin. "Active learning for multiclass
        cost-sensitive classification using probabilistic models." 2013
        Conference on Technologies and Applications of Artificial Intelligence.
        IEEE, 2013.
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
                f"Method `entropy` does not support cost matrices but "
                f"`cost_matrix` was not None."
            )
    else:
        raise ValueError(
            "Supported methods are ['least_confident', 'margin_sampling', "
            "'entropy'], the given one is: {}.".format(method)
        )


def expected_average_precision(classes, probas):
    """
    Calculate the expected average precision.

    Parameters
    ----------
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    probas : np.ndarray, shape=(n_X_cand, n_classes)
        The probabilistic estimation for each classes and all instance in
        candidates.

    Returns
    -------
    score : np.ndarray, shape=(n_X_cand)
        The expected average precision score of all instances in candidates.

    References
    ----------
    [1] Wang, Hanmo, et al. "Uncertainty sampling for action recognition
        via maximizing expected average precision."
        IJCAI International Joint Conference on Artificial Intelligence. 2018.
    """
    # Check if `probas` is valid.
    probas = check_array(
        probas,
        accept_sparse=False,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
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
    for i in range(len(classes)):
        for j in range(len(probas)):
            # The i-th column of p without p[j,i]
            p = probas[:, i]
            p = np.delete(p, [j])
            # Sort p in descending order
            p = np.flipud(np.sort(p, axis=0))

            # calculate g_arr
            g_arr = np.zeros((len(p), len(p)))
            for n in range(len(p)):
                for h in range(n + 1):
                    g_arr[n, h] = _g(n, h, p, g_arr)

            # calculate f_arr
            f_arr = np.zeros((len(p) + 1, len(p) + 1))
            for a in range(len(p) + 1):
                for b in range(a + 1):
                    f_arr[a, b] = _f(a, b, p, f_arr, g_arr)

            # calculate score
            for t in range(len(p)):
                score[j] += f_arr[len(p), t + 1] / (t + 1)

    return score


# g-function for expected_average_precision
def _g(n, t, p, g_arr):
    if t > n or (t == 0 and n > 0):
        return 0
    if t == 0 and n == 0:
        return 1
    return p[n - 1] * g_arr[n - 1, t - 1] + (1 - p[n - 1]) * g_arr[n - 1, t]


# f-function for expected_average_precision
def _f(n, t, p, f_arr, g_arr):
    if t > n or (t == 0 and n > 0):
        return 0
    if t == 0 and n == 0:
        return 1
    return (
            p[n - 1] * f_arr[n - 1, t - 1]
            + p[n - 1] * t * g_arr[n - 1, t - 1] / n
            + (1 - p[n - 1]) * f_arr[n - 1, t]
    )
