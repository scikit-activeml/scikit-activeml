"""
Module implementing 'Value Of Information' active learning strategy.
"""
# Author: Pascal Mergard <pascal.mergard@student.uni-kassel.de>
from copy import deepcopy

import numpy as np
from sklearn import clone

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy, \
    SkactivemlClassifier
from skactiveml.utils import check_cost_matrix, check_type, \
    fit_if_not_fitted, simple_batch, labeled_indices, check_X_y


class VOI(SingleAnnotPoolBasedQueryStrategy):
    """Value Of Information

    This class implements the 'Value Of Information' (VOI) active learning
    strategy as it is defined in [1]. It supports multiclass problems as well
    as different labeling costs for each class and/or each candidate.

    Parameters
    ----------
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost/Risk matrix with C[i,j] defining the cost of predicting class j
        for a sample with the actual class i.
    labeling_cost : scalar or array-like, default=None
        Denote the cost of knowing the class label of an instance from the
        unlabeled set. We assume that the labeling cost and the cost matrix
        are measured with the same currency. `labeling_cost` should be a scalar
        or an array-like with shape (len(X_cand), ), (1, n_classes) or
        (len(X_cand), n_classes).
    ignore_partial_fit : bool, default=False
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.
    random_state : numeric | np.random.RandomState, default=None
        Random state to use.

    References
    ----------
    [1] Kapoor, Ashish, Eric Horvitz, and Sumit Basu. "Selective Supervision:
        Guiding Supervised Learning with Decision-Theoretic Active Learning."
        IJCAI. Vol. 7. 2007.
    """

    def __init__(self, cost_matrix=None, labeling_cost=None,
                 ignore_partial_fit=False, random_state=None):
        super().__init__(random_state=random_state)

        self.labeling_cost = labeling_cost
        self.cost_matrix = cost_matrix
        self.ignore_partial_fit = ignore_partial_fit

    def query(self, X_cand, clf, X, y, sample_weight=None,
              sample_weight_cand=None, batch_size=1, return_utilities=False):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : array-like of shape (n_candidate_samples, n_features)
            Candidate samples from which the strategy can select.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        X: array-like of shape (n_samples, n_features)
            Complete training data set.
        y: array-like, shape (n_samples)
            Labels of the training data set.
        sample_weight : array-like of shape (n_samples), Default=None
            Weights of training samples in `X`.
        sample_weight_cand : array-like of shape (n_candidate_samples),
        Default=None)
            Weights of candidate samples in `X_cand`.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of all candidate samples after each selected
            sample of the batch, e.g., `utilities[0]` indicates the utilities
            used for selecting the first sample (with index `query_indices[0]`)
            of the batch.
        """
        # Validate input parameters.
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        utilities = value_of_information(
            clf=clf,
            X_cand=X_cand,
            X=X,
            y=y,
            cost_matrix=self.cost_matrix,
            labeling_cost=self.labeling_cost,
            sample_weight=sample_weight,
            sample_weight_cand=sample_weight_cand,
            ignore_partial_fit=self.ignore_partial_fit
        )

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def value_of_information(clf, X_cand, X, y, cost_matrix=None,
                         labeling_cost=None, sample_weight=None,
                         sample_weight_cand=None, ignore_partial_fit=False):
    """Compute the 'Value Of Information'.

    This function computes the 'Value Of Information' (VOI) as it is defined
    in [1]. It supports multiclass problems as well as different labeling costs
    for each class and/or each candidate.

    Parameters
    ----------
    clf : skactiveml.base.SkactivemlClassifier
        Model implementing the methods `fit` and `predict_proba`.
    X_cand : array-like of shape (n_candidate_samples, n_features)
        Candidate samples from which the strategy can select.
    X: array-like, shape (n_samples, n_features)
        Complete training data set.
    y: array-like, shape (n_samples)
        Labels of the training data set.
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost/Risk matrix with C[i,j] defining the cost of predicting class j
        for a sample with the actual class i.
    labeling_cost : scalar or array-like, default=None
        Denote the cost of knowing the class label of an instance of the
        unlabeled set. The labeling cost and the cost matrix
        are measured with the same currency. `labeling_cost` should be a scalar
        or an array-like with shape (len(X_cand), ), (1, n_classes) or
        (len(X_cand), n_classes).
    sample_weight : array-like of shape (n_samples), Default=None
        Weights of training samples in `X`.
    sample_weight_cand : array-like, shape (n_candidate_samples),
    Default=None)
        Weights of candidate samples in `X_cand`.
    ignore_partial_fit : bool, default=False
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.

    Returns
    -------
    voi : np.ndarray of shape (n_candidates)
        The value of information of all unlabeled instances.

    References
    ----------
    [1] Kapoor, Ashish, Eric Horvitz, and Sumit Basu. "Selective Supervision:
        Guiding Supervised Learning with Decision-Theoretic Active Learning."
        IJCAI. Vol. 7. 2007.
    """
    # Validate classifier type.
    check_type(clf, 'clf', SkactivemlClassifier)

    # Check X, y, sample_weight, sample_weight_cand.
    X, y, X_cand, sample_weight, sample_weight_cand = check_X_y(
        X, y, X_cand, sample_weight, sample_weight_cand,
        missing_label=clf.missing_label
    )

    # Check whether to use `fit` or `partial_fit`.
    check_type(ignore_partial_fit, 'ignore_partial_fit', bool)
    use_fit = ignore_partial_fit or not hasattr(clf, 'partial_fit')

    # Fit the classifier.
    clf = fit_if_not_fitted(clf, X, y, sample_weight)

    # Check n_classes
    n_classes = len(clf.classes_)
    # Check the cost matrix.
    cost_matrix = 1 - np.eye(len(clf.classes_)) if cost_matrix is None else \
        cost_matrix
    cost_matrix = check_cost_matrix(cost_matrix, n_classes=n_classes,
                                    diagonal_is_zero=True)
    # Extract the labeled indices.
    X_labeled = X[labeled_indices(y)]
    y_labeled = np.array(y[labeled_indices(y)], dtype=int)

    # Get the predicted probabilities from the classifier.
    if len(X_labeled) == 0:
        probas_X = np.empty(shape=(0, clf.n_features_in_))
    else:
        probas_X = clf.predict_proba(X_labeled)
    probas_X_cand = clf.predict_proba(X_cand)

    # Validate labeling_cost.
    if labeling_cost is None:
        labeling_cost = 1
    labeling_cost = np.asarray(labeling_cost, dtype=float)
    if labeling_cost.shape == ():
        labeling_cost_cand = labeling_cost.reshape(-1)
    elif labeling_cost.shape == (1,) \
            or labeling_cost.shape == (len(X_cand),):
        labeling_cost_cand = labeling_cost
    elif labeling_cost.shape == (1, n_classes) \
            or labeling_cost.shape == (len(X_cand), n_classes):
        labeling_cost_cand = np.sum(probas_X_cand*labeling_cost, axis=1)
    else:
        raise ValueError("'labeling_cost' should be a scalar or an array-like "
                         "with shape (len(X_cand), ), (1, n_classes) or "
                         "(len(X_cand), n_classes). The given shape is "
                         f"{labeling_cost.shape}")

    # Compute the total risk on the data.
    totalRisk = _total_unlabeled_risk(probas_X_cand, cost_matrix) + \
                _total_labeled_risk(y_labeled, probas_X, cost_matrix)

    # Compute the expected total risk on the unlabeled data.
    expected_risk_per_class = np.empty((len(X_cand), n_classes))
    clf_refit = clone(clf).fit if use_fit else deepcopy(clf).partial_fit
    for j, x in enumerate(X_cand):
        for yi in range(n_classes):
            # Create sample array for the retraining of the classifier.
            X_new = np.vstack((X, [x])) if use_fit else np.array([x])
            X_labeled_new = np.append(X_labeled, [x], axis=0)
            # Create label array for the retraining of the classifier.
            y_new = np.append(y, [[yi]]) if use_fit else np.array([yi])
            y_labeled_new = np.append(y_labeled, yi)
            # Create sample weight for the refit.
            if use_fit:
                sample_weight_cand_new = np.append(sample_weight,
                                                   [sample_weight_cand[j]])
            else:
                sample_weight_cand_new = [sample_weight_cand[j]]
            # Retrain classifier.
            clf_new = clf_refit(X_new, y_new, sample_weight_cand_new)
            # Create a candidate set without candidate j.
            X_cand_new = np.delete(X_cand, j, axis=0)
            # Predict the new probabilities.
            if len(X_cand_new) == 0:
                probas_X_cand_new = np.empty(shape=(0, clf_new.n_features_in_))
            else:
                probas_X_cand_new = clf_new.predict_proba(X_cand_new)
            probas_X = clf_new.predict_proba(X_labeled_new)
            # Compute the expected total risk per class.
            expected_risk_per_class[j, yi] = \
                _total_unlabeled_risk(probas_X_cand_new, cost_matrix) + \
                _total_labeled_risk(y_labeled_new, probas_X, cost_matrix)
    # Compute the expected total risk.
    expected_total_risk = np.sum(probas_X_cand*expected_risk_per_class, axis=1)

    return totalRisk - expected_total_risk - labeling_cost_cand


def _total_labeled_risk(y_labeled, probas, cost_matrix):
    if len(y_labeled) == 0:
        return 0
    R = np.dot(probas, cost_matrix.T)
    return np.sum(np.choose(y_labeled, R.T))


def _total_unlabeled_risk(probas, cost_matrix):
    result = 0.0
    for i, p in enumerate(probas):
        result += np.sum(p * (p * cost_matrix.T).T)
    return result
