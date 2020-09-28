import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array

from skactiveml.base import PoolBasedQueryStrategy
from skactiveml.utils import rand_argmax, is_labeled, MISSING_LABEL


class ExpectedErrorReduction(PoolBasedQueryStrategy):
    """ExpectedErrorReduction
    This class implements the expected error reduction algorithm with different
    loss functions:
     - log loss (log_loss) [1],
     - expected misclassification risk (emr) [2],
     - and cost-sensitive learning (csl) [2].

    Parameters
    ----------
    clf: model to be trained
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


    Attributes
    ----------
    clf: model to be trained
        Model implementing the methods 'fit' and and 'predict_proba'.
    classes: array-like, shape (n_classes)
        List of all possible classes. Must correspond to the classes of clf.
    method: {'log_loss', 'emr', 'csl'}
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.
    C: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i.
        Only supported for least confident variant.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.
    missing_label: str | numeric
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

    def __init__(self, clf, classes, method=EMR, C=None, random_state=None,
                 missing_label=MISSING_LABEL, **kwargs):
        super().__init__(random_state=random_state)

        self.clf = clf
        if getattr(self.clf, 'fit', None) is None or\
                getattr(self.clf, 'predict_proba', None) is None:
            raise TypeError("'clf' must implement the methods 'fit' and "
                            "'predict_proba'")

        self.classes = classes
        if not np.array_equal(clf.classes, self.classes):
            raise ValueError("The given classes are not the same as in the "
                             "classifier.")

        self.method = method
        if self.method not in [ExpectedErrorReduction.EMR,
                               ExpectedErrorReduction.CSL,
                               ExpectedErrorReduction.LOG_LOSS]:
            raise ValueError(
                f"supported methods are [{ExpectedErrorReduction.EMR}, "
                f"{ExpectedErrorReduction.CSL}, "
                f"{ExpectedErrorReduction.LOG_LOSS}], the given one is: "
                f"{self.method}"
            )

        if C is None:
            self.C = 1 - np.eye(len(self.classes))
        else:
            self.C = check_array(C)
        if np.size(self.C, axis=0) != np.size(self.C, axis=1):
            raise ValueError("C must be a square matrix")
        if np.size(self.C, axis=0) != np.size(classes):
            raise ValueError("C must be a (n_classes, n_classes) matrix")

        self.random_sate = random_state

        self.missing_label = missing_label

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like (n_candidates, n_features)
            Unlabeled candidate samples
        X: array-like (n_samples, n_features)
            Complete data set
        y: array-like (n_samples)
            Labels of the data set
        return_utilities: bool (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        selection: np.ndarray, shape (1)
            The index of the queried instance.
        utilities: np.ndarray shape (1, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """

        X_cand = check_array(X_cand, force_all_finite=False)
        X = np.array(X)
        y = np.array(y)
        labeled_indices = is_labeled(y, missing_label=self.missing_label)
        X_labeled = X[labeled_indices]
        y_labeled = y[labeled_indices]

        # calculate the utilities
        utilities = expected_error_reduction(clf=self.clf, X_labeled=X_labeled,
                                             y_labeled=y_labeled,
                                             X_unlabeled=X_cand,
                                             classes=self.classes,
                                             C=self.C, method=self.method)

        query_indices = rand_argmax([utilities], axis=1,
                                    random_state=self.random_state)
        if return_utilities:
            return query_indices, np.array([utilities])
        else:
            return query_indices


def expected_error_reduction(clf, X_labeled, y_labeled, X_unlabeled, classes,
                             C, method='emr'):
    """
    Computes least confidence as uncertainty scores. In case of a given cost
    matrix C, maximum expected cost is implemented as score.

    Parameters
    ----------
    clf: sklearn classifier with predict_proba method
        Model whose expected error reduction is measured.
    X_labeled: array-like, shape (n_labeled_samples, n_features)
        Labeled samples.
    y_labeled: array-like, shape (n_labeled_samples)
        Class labels of labeled samples.
    X_unlabeled: array-like, shape (n_unlabeled_samples)
        Unlabeled samples.
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

    if X_labeled.shape[0] > 0 and X_unlabeled.shape[0] > 0 and \
            not X_labeled.shape[1] == X_unlabeled.shape[1]:
        raise ValueError("X_labeled and X_unlabeled must have the same number "
                         "of features.")
    clf = clone(clf)
    clf.fit(X_labeled, y_labeled)
    # if not np.array_equal(clf.classes_, classes):
    #     raise ValueError("The given classes are not the same as in the "
    #                      "classifier.")

    n_classes = len(classes)
    P = clf.predict_proba(X_unlabeled)
    C = 1 - np.eye(np.size(P, axis=1)) if C is None else C
    errors = np.zeros(len(X_unlabeled))
    errors_per_class = np.zeros(n_classes)
    for i, x in enumerate(X_unlabeled):
        for yi in range(n_classes):
            clf.fit(np.vstack((X_labeled, [x])), np.append(y_labeled, [[yi]]))
            if method == 'emr':
                P_new = clf.predict_proba(X_unlabeled)
                costs = np.sum((P_new.T[:, None] * P_new.T).T * C)
            elif method == 'csl':
                if len(X_labeled) > 0:
                    costs = np.sum(clf.predict_proba(X_labeled) * C[y_labeled])
                else:
                    costs = 0
            elif method == 'log_loss':
                P_new = clf.predict_proba(X_unlabeled)
                costs = -np.sum(P_new * np.log(P_new))
            else:
                raise ValueError(f"supported methods are ['emr', 'csl'], the "
                                 f"given one " "is: {method}")
            errors_per_class[yi] = P[i, yi] * costs
        errors[i] = errors_per_class.sum()
    # print(P)
    return -errors
