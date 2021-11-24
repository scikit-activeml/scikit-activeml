"""
Module implementing 'Selective Supervision' active learning strategy.
"""
# Author: Pascal Mergard <pascal.mergard@student.uni-kassel.de>
from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.utils import check_array

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy, \
    SkactivemlClassifier
from skactiveml.utils import check_cost_matrix, check_classes, check_type, \
    fit_if_not_fitted, simple_batch, labeled_indices, unlabeled_indices, \
    check_scalar


class SelectiveSupervision(SingleAnnotPoolBasedQueryStrategy):
    # TODO docstring for SelectiveSupervision

    def __init__(self, cost=None, cost_matrix=None, ignore_partial_fit=False,
                 random_state=None):
        super().__init__(random_state=random_state)

        self.cost = cost
        self.cost_matrix = cost_matrix
        self.ignore_partial_fit = ignore_partial_fit

    def query(self, X_cand, clf, X, y, sample_weight=None,
              batch_size=1, return_utilities=False):
        # TODO docstring for SelectiveSupervision.query
        # Validate input parameters.
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        utilities = value_of_information()

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def value_of_information(clf, X_cand, X, y, cost_matrix=None,
                         labeling_cost=None, ignore_partial_fit=False):
    """Compute value of information.

    This function computes the 'value of information' (VOI) as it is defined
    in [1].

    Parameters
    ----------
    clf : skactiveml.base.SkactivemlClassifier
        Model implementing the methods `fit` and `predict_proba`.
    X_cand : array-like, shape (n_candidate_samples, n_features)
        Candidate samples from which the strategy can select.
    X: array-like, shape (n_samples, n_features)
        Complete training data set.
    y: array-like, shape (n_samples)
        Labels of the training data set.
    cost_matrix : array-like, shape (n_classes, n_classes), default=None
        Cost/Risk matrix with C[i,j] defining the cost of predicting class j
        for a sample with the actual class i.
    labeling_cost : scalar or array-like, default=None
        Denote the cost of knowing the class label of an instance of the
        unlabeled set. We assume that the labeling cost and the cost matrix
        are measured with the same currency. Valide shapes are (1, ),
        (len(X_cand), 1), (1, n_classes) or (len(X_cand), n_classes).
    ignore_partial_fit : bool, default=False
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.

    Returns
    -------
    voi : np.ndarray, shape (n_candidates)
        The value of information of all unlabeled instances.

    References
    ----------
    [1] Kapoor, Ashish, Eric Horvitz, and Sumit Basu. "Selective Supervision:
        Guiding Supervised Learning with Decision-Theoretic Active Learning."
        IJCAI. Vol. 7. 2007.
    """
    # Validate classifier type.
    check_type(clf, SkactivemlClassifier, 'clf')

    # Check whether to use `fit` or `partial_fit`.
    check_type(ignore_partial_fit, bool, 'ignore_partial_fit')
    use_fit = ignore_partial_fit or not hasattr(clf, 'partial_fit')

    # Fit the classifier.
    clf = fit_if_not_fitted(clf, X, y)

    # Check the cost matrix.
    n_classes = len(clf.classes_)
    if n_classes != 2:
        raise ValueError(f'The number of classes must be 2, but {n_classes} '
                         f'were given.')
    cost_matrix = 1 - np.eye(len(clf.classes_)) if cost_matrix is None else \
        cost_matrix
    cost_matrix = check_cost_matrix(cost_matrix, n_classes=n_classes,
                                    diagonal_is_zero=True)
    # Extract the labeled indices.
    X_labeled = X[labeled_indices(y)]
    y_labeled = y[labeled_indices(y)]

    # Get the predicted probabilities from the classifier.
    probas_X = clf.predict_proba(X_labeled)
    probas = clf.predict_proba(X_cand)

    # Validate labeling_cost.
    labeling_cost = np.asarray(labeling_cost)
    if labeling_cost is None:
        labeling_cost_cand = 1
    elif labeling_cost.shape == (1,) \
            or labeling_cost.shape == (len(X_cand), 1):
        labeling_cost_cand = labeling_cost
    elif labeling_cost.shape == (1, n_classes) \
            or labeling_cost.shape == (len(X_cand), n_classes):
        labeling_cost_cand = np.sum(probas*labeling_cost, axis=1)
    else:
        raise ValueError("'labeling_cost' should be a scalar or an array-like "
                         "with shape (len(X_cand), 1), (1, n_classes) or "
                         "(len(X_cand), n_classes). The given shape is "
                         f"{labeling_cost.shape}")

    # Compute the total risk on the data.
    totalRisk = _total_unlabeld_risk(probas, cost_matrix) + \
                _total_labeled_risk(y_labeled, probas_X, cost_matrix)

    # Compute the expected total risk on the unlabeled data.
    expected_risk_per_class = np.empty((len(X_cand), n_classes))
    clf_refit = clone(clf).fit if use_fit else deepcopy(clf).partial_fit
    for j, x in enumerate(X_cand):
        for yi in range(n_classes):
            # Create sample array for the retraining of the classifier.
            X_new = np.vstack((X, [x])) if use_fit else np.array([x])
            # Create label array for the retraining of the classifier.
            y_new = np.append(y, [[yi]]) if use_fit else np.array([yi])
            # Retrain classifier.
            clf_new = clf_refit(X_new, y_new)
            # Create a candidate set without candidate j.
            X_cand_new = np.delete(X_cand, j)
            # Predict the new probabilities.
            probas_X_cand = clf_new.predict_proba(X_cand_new)
            probas_X = clf_new.predict_proba(X_new)
            # Compute the expected total risk per class.
            expected_risk_per_class[j, yi] = \
                _total_unlabeld_risk(probas_X_cand, cost_matrix) + \
                _total_labeled_risk(y_labeled, probas_X, cost_matrix)
    # Compute the expected total risk.
    expected_total_risk = np.sum(probas*expected_risk_per_class, axis=1)

    return totalRisk - expected_total_risk - labeling_cost_cand


def _total_labeled_risk(y_labeled, probas, cost_matrix):
    R = probas * cost_matrix.T
    return np.sum(np.where[y_labeled == 1, R[0], R[1]])


def _total_unlabeld_risk(probas, cost_matrix):
    # TODO adapt for multiclass problem
    return np.sum(cost_matrix.sum() * probas[0] * probas[1])
