import copy
import numpy as np

from sklearn.utils import check_array

from ..base import PoolBasedQueryStrategy
from ..utils import rand_argmax, is_labeled


class ExpectedErrorReduction(PoolBasedQueryStrategy):
    """ExpectedErrorReduction
    This class implements the expected error reduction algorithm with different loss functions:
     - log loss (log_loss) [1],
     - expected misclassification risk (emr) [2],
     - and cost-sensitive learning (csl) [2].

    Parameters
    ----------
    model: model to be trained
        Model implementing the methods 'fit' and and 'predict_proba'.
    method_: {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is cost-insensitive, while 'emr' and 'csl' are
        cost-sensitive variants.
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    C: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for least confident variant.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    model_: model to be trained
        Model implementing the methods 'fit' and and 'predict_proba'.
    method_: {'log_loss', 'emr', 'csl'}
        Variant of expected error reduction to be used: 'log_loss' is cost-insensitive, while 'emr' and 'csl' are
        cost-sensitive variants.
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    C_: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a sample which actually belongs
        to class i.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Margineantu, D. D. (2005, July). Active cost-sensitive learning. In IJCAI (Vol. 5, pp. 1622-1623).
    """

    EMR = 'emr'
    CSL = 'csl'
    LOG_LOSS = 'log_loss'

    def __init__(self, clf, classes, method=EMR, C=None, random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)

        self.clf = clf
        if getattr(self.clf, 'fit', None) is None or getattr(self.clf, 'predict_proba', None) is None:
            raise TypeError(
                "'model' must implement the methods 'fit' and 'predict_proba'"
            )

        self.C_ = C
        if self.C_ is not None:
            self.C_ = check_array(self.C_)
            if np.size(self.C_, axis=0) != np.size(self.C_, axis=1):
                raise ValueError(
                    "C must be a square matrix"
                )

        self.method_ = method
        if self.method_ not in [ExpectedErrorReduction.EMR, ExpectedErrorReduction.CSL]:
            raise ValueError(
                "supported methods are [{}, {}, {}], the given one " "is: {}".format(
                    ExpectedErrorReduction.EMR, ExpectedErrorReduction.CSL,
                    ExpectedErrorReduction.LOG_LOSS, self.method_)
            )

        self.classes_ = classes

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like (n_candidates, n_features)
            Unlabeled candidate samples
        X: array-like (n_training_samples, n_features)
            Complete data set
        y: array-like (n_training_samples)
            Labels of the data set
        return_utilities: bool (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        selection: np.ndarray, shape (1)
            The index of the queried instance.
        utilities: np.ndarray shape (1, n_candidates)
            The utilities of all instances in X_cand (only if return_utilities=True).
        """

        X_cand = check_array(X_cand, force_all_finite=False)
        X = np.array(X)
        y = np.array(y)
        labeled_indices = is_labeled(y)
        X_labeled = X[labeled_indices]
        y_labeled = y[labeled_indices]

        # caculate the utilities
        utilities = expected_error_reduction(clf=self.clf, X_labeled=X_labeled, y_labeled=y_labeled,
                                        X_unlabeled=X_cand, classes=self.classes_, C=self.C_,
                                        method=self.method_)

        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


def expected_error_reduction(clf, X_labeled, y_labeled, X_unlabeled, classes, C=None, method='emr'):
    """
    Computes least confidence as uncertainty scores. In case of a given cost matrix C,
    maximum expected cost is implemented as score.

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
        Cost matrix with C[i,j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for least confident variant.
    method: {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is cost-insensitive, while 'emr' and 'csl' are
        cost-sensitive variants.
    """

    if not X_labeled.shape[1] == X_unlabeled.shape[1]:
        raise ValueError("X_labeled and X_unlabeled must have the same number of features.")
    clf = copy.deepcopy(clf)
    clf.fit(X_labeled, y_labeled)
    if not np.array_equal(clf.classes_, classes):
        raise ValueError("The given classes are not the same as in the classifier.")

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
                raise ValueError(
                    "supported methods are ['emr', 'csl'], the given one " "is: {}".format(method)
                )
            errors_per_class[yi] = P[i, yi] * costs
        errors[i] = errors_per_class.sum()
    return -errors
