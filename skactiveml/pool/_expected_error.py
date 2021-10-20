from copy import deepcopy

import numpy as np
from sklearn.base import clone

from ..base import SingleAnnotPoolBasedQueryStrategy, SkactivemlClassifier
from ..utils import check_type, is_labeled, simple_batch, \
    fit_if_not_fitted, check_cost_matrix, check_X_y


class ExpectedErrorReduction(SingleAnnotPoolBasedQueryStrategy):
    """Expected Error Reduction

    This class implements the expected error reduction algorithm with different
    loss functions:
     - log loss (log_loss) [1],
     - expected misclassification risk (emr) [2],
     - and cost-sensitive learning (csl) [3].

    Parameters
    ----------
    method: {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.
    cost_matrix: array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Only supported for least confident
        variant.
    ignore_partial_fit: bool, optional (default=False)
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for annotator selection.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. (2009, June).
        Multi-class active learning for image classification.
        In 2009 IEEE Conference on Computer Vision and Pattern Recognition
        (pp. 2372-2379). IEEE.
    [3] Margineantu, D. D. (2005, July). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """

    EMR = 'emr'
    CSL = 'csl'
    LOG_LOSS = 'log_loss'

    def __init__(self, method=EMR, cost_matrix=None, ignore_partial_fit=False,
                 random_state=None):
        super().__init__(random_state=random_state)
        self.method = method
        self.cost_matrix = cost_matrix
        self.ignore_partial_fit = ignore_partial_fit

    def query(self, X_cand, clf, X=None, y=None, sample_weight=None,
              sample_weight_cand=None, batch_size=1, return_utilities=False):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand : array-like, shape (n_candidate_samples, n_features)
            Candidate samples from which the strategy can select.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        X : array-like, shape (n_samples, n_features), optional (default=None)
            Complete training data set.
        y : array-like, shape (n_samples), optional (default=None)
            Labels of the training data set.
        sample_weight : array-like, shape (n_samples), optional
        (default=None)
            Weights of training samples in `X`.
        sample_weight_cand : array-like, shape (n_candidate_samples), optional
        (default=None)
            Weights of candidate samples in `X_cand`.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
        utilities : numpy.ndarray, shape (batch_size, n_samples)
            The utilities of all candidate samples after each selected
            sample of the batch, e.g., `utilities[0]` indicates the utilities
            used for selecting the first sample (with index `query_indices[0]`)
            of the batch.
        """
        # Validate input parameters.
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        # Calculate utilities
        utilities = expected_error_reduction(
            clf=clf, X_cand=X_cand, X=X, y=y, cost_matrix=self.cost_matrix,
            method=self.method, sample_weight=sample_weight,
            sample_weight_cand=sample_weight_cand,
            ignore_partial_fit=self.ignore_partial_fit
        )

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def expected_error_reduction(clf, X_cand, X=None, y=None, cost_matrix=None,
                             method='emr', sample_weight_cand=None,
                             sample_weight=None, ignore_partial_fit=False):
    """Compute uncertainty scores.

    In case of a given cost matrix C, maximum expected cost is implemented as
    score.

    Parameters
    ----------
    clf : skactiveml.base.SkactivemlClassifier
        Model implementing the methods `fit` and `predict_proba`.
    X_cand : array-like, shape (n_candidate_samples, n_features)
        Candidate samples from which the strategy can select.
    X : array-like, shape (n_samples, n_features), optional (default=None)
        Complete training data set.
    y : array-like, shape (n_samples), optional (default=None)
        Labels of the training data set.
    cost_matrix : array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Only supported for least confident
        variant.
    method : {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.
    sample_weight : array-like, shape (n_samples), optional
    (default=None)
        Weights of training samples in `X`.
    sample_weight_cand : array-like, shape (n_candidate_samples), optional
    (default=None)
        Weights of candidate samples in `X_cand`.
    ignore_partial_fit : bool, optional (default=False)
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.

    Returns
    -------
    utilities : np.ndarray, shape (n_candidates)
        The utilities of all unlabeled instances.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. (2009, June).
        Multi-class active learning for image classification.
        In 2009 IEEE Conference on Computer Vision and Pattern Recognition
        (pp. 2372-2379). IEEE.
    [3] Margineantu, D. D. (2005, July). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """
    # Check if the classifier and its arguments are valid.
    check_type(clf, SkactivemlClassifier, 'clf')

    # Check whether to use `fit` or `partial_fit`.
    check_type(ignore_partial_fit, bool, 'ignore_partial_fit')
    use_fit = ignore_partial_fit or not hasattr(clf, 'partial_fit')
    if use_fit and (X is None or y is None):
        raise ValueError(
            '`X` and `y` cannot be None for a classifier using `fit` for '
            'retraining.'
        )
    if (X is None or y is None) and method == 'csl':
        raise ValueError(
            "`X` and `y` cannot be None for `method='csl'`."
        )
    use_sample_weight = sample_weight is not None \
                        or sample_weight_cand is not None
    if use_fit and (bool(sample_weight is None)
                    != bool(sample_weight_cand is None)):
        raise ValueError(
            '`sample_weight` and `sample_weight_cand` must either both be '
            'None or array-like, if the fit method is used.'
        )
    X, y, X_cand, sample_weight, sample_weight_cand = check_X_y(
        X, y, X_cand, sample_weight, sample_weight_cand,
        force_all_finite=False, missing_label=clf.missing_label
    )

    # Refit classifier.
    if use_sample_weight:
        clf = fit_if_not_fitted(clf, X, y, sample_weight, False)
    else:
        clf = fit_if_not_fitted(clf, X, y, None, False)
    clf_refit = clone(clf).fit if use_fit else deepcopy(clf).partial_fit

    # Check cost matrix.
    n_classes = len(clf.classes_)
    cost_matrix = 1 - np.eye(len(clf.classes_)) if cost_matrix is None else \
        cost_matrix
    cost_matrix = check_cost_matrix(cost_matrix, n_classes)

    # Compute class-membership probabilities of candidate samples.
    P = clf.predict_proba(X_cand)

    # Storage for computed errors per candidate sample.
    errors = np.zeros(len(X_cand))
    errors_per_class = np.zeros(n_classes)

    # Iterate over candidate samples
    for i, x in enumerate(X_cand):
        # Simulate acquisition of label for each candidate sample and class.
        for yi in range(n_classes):
            # Create sample array for the retraining of the classifier.
            X_new = np.vstack((X, [x])) if use_fit else np.array([x])
            # Create label array for the retraining of the classifier.
            y_new = np.append(y, [[yi]]) if use_fit else np.array([yi])
            # Check whether sample weights are used.
            if use_sample_weight:
                # Create sample weight array for the retraining of the
                # classifier.
                w = sample_weight_cand[i]
                if use_fit:
                    sample_weight_new = np.append(sample_weight, [[w]])
                else:
                    sample_weight_new = np.array([w])
                # Retrain classifier with sample weights.
                clf_new = clf_refit(X_new, y_new, sample_weight_new)
            else:
                # Retrain classifier without sample weights.
                clf_new = clf_refit(X_new, y_new)
            if method == 'emr':
                P_new = clf_new.predict_proba(X_cand)
                costs = np.sum((P_new.T[:, None] * P_new.T).T * cost_matrix)
            elif method == 'csl':
                is_lbld = is_labeled(y, clf_new.missing_label)
                X_labeled = X[is_lbld]
                y_labeled = y[is_lbld]
                y_indices = [np.where(clf_new.classes_ == label)[0][0]
                             for label in y_labeled]
                if len(X_labeled) > 0:
                    costs = np.sum(
                        clf_new.predict_proba(X_labeled) *
                        cost_matrix[y_indices]
                    )
                else:
                    costs = 0
            elif method == 'log_loss':
                P_new = clf_new.predict_proba(X_cand)
                costs = -np.sum(P_new * np.log(P_new + np.finfo(float).eps))
            else:
                raise ValueError(
                    f"Supported methods are [{ExpectedErrorReduction.EMR}, "
                    f"{ExpectedErrorReduction.CSL}, "
                    f"{ExpectedErrorReduction.LOG_LOSS}], the given one is: "
                    f"{method}"
                )
            errors_per_class[yi] = P[i, yi] * costs
        errors[i] = errors_per_class.sum()
    return -errors
