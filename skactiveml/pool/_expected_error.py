import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.base import ClassFrequencyEstimator
from skactiveml.utils import check_classifier_params, check_scalar, check_X_y
from skactiveml.utils import rand_argmax, MISSING_LABEL, is_labeled


class ExpectedErrorReduction(SingleAnnotPoolBasedQueryStrategy):
    """Expected Error Reduction.

    This class implements the expected error reduction algorithm with different
    loss functions:
     - log loss (log_loss) [1],
     - expected misclassification risk (emr) [2],
     - and cost-sensitive learning (csl) [2].

    Parameters
    ----------
    clf : ClassFrequencyEstimator
        Model implementing the methods 'fit' and and 'predict_proba'.
    classes: array-like, shape (n_classes)
        List of all possible classes. Must correspond to the classes of clf.
    method: {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.
    C: array-like, shape (n_classes, n_classes), optional (default=None)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i. Only supported for least confident
        variant.
    random_state: numeric | np.random.RandomState, optional (defatult=None)
        Random state for annotator selection.
    missing_label: str | numeric, optional (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Margineantu, D. D. (2005, July). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """

    EMR = 'emr'
    CSL = 'csl'
    LOG_LOSS = 'log_loss'

    def __init__(self, clf, classes, method=EMR, cost_matrix=None,
                 random_state=None, missing_label=MISSING_LABEL):
        super().__init__(random_state=random_state)
        self.clf = clf
        self.classes = classes
        self.method = method
        self.cost_matrix = cost_matrix
        self.missing_label = missing_label

    def query(self, X_cand, X, y, batch_size=1, return_utilities=False,
              **kwargs):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples
        X: array-like, shape (n_samples, n_features)
            Complete data set
        y: array-like, shape (n_samples)
            Labels of the data set
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        return_utilities: bool, optional (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (batch_size)
            The index of the queried instance.
        utilities: np.ndarray, shape (batch_size, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        # Set the cost matrix to the default value if it is not given
        if self.cost_matrix is None:
            self.cost_matrix = 1 - np.eye(len(self.classes))

        # Check if the classifier and its arguments are valid
        if not isinstance(self.clf, ClassFrequencyEstimator):
            raise TypeError("'clf' must implement methods according to "
                            "'ClassFrequencyEstimator'.")
        check_classifier_params(self.classes, self.missing_label,
                                self.cost_matrix)

        # Check if the given classes are the same
        if not np.array_equal(self.clf.classes, self.classes):
            raise ValueError("The given classes are not the same as in the "
                             "classifier.")

        # Check if a valid method is given
        if self.method not in [ExpectedErrorReduction.EMR,
                               ExpectedErrorReduction.CSL,
                               ExpectedErrorReduction.LOG_LOSS]:
            raise ValueError(
                f"supported methods are [{ExpectedErrorReduction.EMR}, "
                f"{ExpectedErrorReduction.CSL}, "
                f"{ExpectedErrorReduction.LOG_LOSS}], the given one is: "
                f"{self.method}"
            )

        # Check the given data
        X_cand = check_array(X_cand, force_all_finite=False)
        X, y = check_X_y(X, y, force_all_finite=False,
                         missing_label=self.missing_label)

        # Check 'batch_size'
        check_scalar(batch_size, 'batch_size', int, min_val=1)

        # Calculate utilities and return the output
        utilities = _expected_error_reduction(self.clf, X_cand, X, y,
                                              self.classes, self.cost_matrix,
                                              self.method)
        query_indices = rand_argmax(utilities, self.random_state)
        if return_utilities:
            return query_indices, np.array([utilities])
        else:
            return query_indices


def _expected_error_reduction(clf, X_cand, X, y, classes, C, method='emr'):
    """Compute least confidence as uncertainty scores.

    In case of a given cost matrix C, maximum expected cost is implemented as
    score.

    Parameters
    ----------
    clf: sklearn classifier with predict_proba method
        Model whose expected error reduction is measured.
    X_cand: array-like, shape (n_candidates, n_features)
        Unlabeled candidate samples
    X: array-like, shape (n_samples, n_features)
        Complete data set
    y: array-like, shape (n_samples)
        Labels of the data set
    classes: array-like, shape (n_classes)
        List of classes.
    C: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i.
        Only supported for least confident variant.
    method: {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.

    Returns
    -------
    utilities: np.ndarray, shape (n_unlabeled_samples)
        The utilities of all unlabeled instances.
    """
    # Check if 'X' and 'X_cand' have the same number of features
    if X.shape[0] > 0 and X_cand.shape[0] > 0 and \
            not X.shape[1] == X_cand.shape[1]:
        raise ValueError("X and X_cand must have the same number "
                         "of features.")

    clf = clone(clf)
    clf.fit(X, y)

    n_classes = len(classes)
    P = clf.predict_proba(X_cand)
    C = 1 - np.eye(np.size(P, axis=1)) if C is None else C
    errors = np.zeros(len(X_cand))
    errors_per_class = np.zeros(n_classes)
    for i, x in enumerate(X_cand):
        for yi in range(n_classes):
            clf.fit(np.vstack((X, [x])), np.append(y, [[yi]]))
            if method == 'emr':
                P_new = clf.predict_proba(X_cand)
                costs = np.sum((P_new.T[:, None] * P_new.T).T * C)
            elif method == 'csl':
                labeled_indices = is_labeled(y, clf.missing_label)
                X_labeled = X[labeled_indices]
                y_labeled = y[labeled_indices]
                y_indices = [np.where(classes == label)[0][0]
                             for label in y_labeled]
                if len(X_labeled) > 0:
                    costs = np.sum(clf.predict_proba(X_labeled) * C[y_indices])
                else:
                    costs = 0
            elif method == 'log_loss':
                P_new = clf.predict_proba(X_cand)
                costs = -np.sum(P_new * np.log(P_new + np.finfo(float).eps))
            else:
                raise ValueError(f"supported methods are ['emr', 'csl'], the "
                                 f"given one " "is: {method}")
            errors_per_class[yi] = P[i, yi] * costs
        errors[i] = errors_per_class.sum()
    return -errors
