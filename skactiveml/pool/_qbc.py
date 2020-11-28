
import numpy as np
import warnings

from sklearn import clone
from sklearn.base import BaseEstimator

from ..base import SingleAnnotPoolBasedQueryStrategy

from sklearn.ensemble import BaggingClassifier, BaseEnsemble
from sklearn.utils import check_array, check_random_state

from ..classifier import SklearnClassifier
from ..utils import MISSING_LABEL, check_X_y, check_scalar, \
    simple_batch, check_classifier_params, check_classes


class QBC(SingleAnnotPoolBasedQueryStrategy):
    """QBC

    The Query-By-Committee (QBC) algorithm minimizes the version space, which
    is the set of hypotheses that are consistent with the current labeled
    training data. This class implement the query-by-bagging method, which uses
    the bagging in sklearn to construct the committee. So your model should be
    a sklearn model.

    Parameters
    ----------
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    clf : sklearn classifier | ensamble
        If clf is an ensemble, it will used as committee. If clf is a
        classifier, it will used for ensemble construction with the specified
        ensemble or with BaggigngClassifier, if ensemble is None. clf must
        implementing the methods 'fit', 'predict'(for vote entropy) and
        'predict_proba'(for KL divergence).
    ensemble : sklearn.ensemble, default=None
        sklear.ensemble used as committee. If None, baggingClassifier is used.
    method : string, default='KL_divergence'
        The method to calculate the disagreement.
        'vote_entropy' or 'KL_divergence' are possible.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.
    **kwargs :
        will be passed on to the ensemble.

    Attributes
    ----------
    ensemble : sklearn.ensemble
        Ensemble used as committee. Implementing the methods 'fit',
        'predict'(for vote entropy) and 'predict_proba'(for KL divergence).
    method : string, default='KL_divergence'
        The method to calculate the disagreement. 'vote_entropy' or
        'KL_divergence' are possible.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    unlabeled_class : scalar | str | None | np.nan, default=np.nan
        Symbol to represent a missing label. Important: We do not differ
        between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.
    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and
        bagging. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 1-9. Morgan Kaufmann, 1998.
    """

    def __init__(self, clf, ensemble=None, method='KL_divergence',
                 classes=None, missing_label=MISSING_LABEL, random_state=None,
                 **kwargs_ensemble):
        super().__init__(random_state=random_state)

        self.missing_label = missing_label
        self.method = method
        self.ensemble = ensemble
        self.clf = clf
        self.classes = classes
        self.kwargs_ensemble = kwargs_ensemble

    def query(self, X_cand, X, y, batch_size=1, return_utilities=False,
              **kwargs):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : array-like
            The unlabeled pool from which to choose.
        X : array-like
            The labeled pool used to fit the classifier.
        y : array-like
            The labels of the labeled pool X.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool (default=False)
            If True, the utilities are returned.

        Returns
        -------
        best_indices : np.ndarray, shape (batch_size)
            The index of the queried instance.
        batch_utilities : np.ndarray,  shape (batch_size, len(X_cnad))
            The utilities of all instances of
            X_cand(if return_utilities=True).
        """
        # Check if the attribute clf is valid
        if not isinstance(self.clf, BaseEstimator):
            raise TypeError("'clf' has to be from type BaseEstimator. "
                            "The given type is {}".format(type(self.clf)))

        self._clf = clone(self.clf)

        # Set and check random state.
        random_state = check_random_state(self.random_state)

        # Check if the classifier and its arguments are valid
        check_classifier_params(self.classes, self.missing_label)

        # check X, y and X_cand
        X, y, X_cand = check_X_y(X, y, X_cand, force_all_finite=False)

        # Check if the batch_size argument is valid.
        check_scalar(batch_size, target_type=int, name='batch_size',
                     min_val=1)
        if len(X_cand) < batch_size:
            warnings.warn(
                "'batch_size={}' is larger than number of candidate samples "
                "in 'X_cand'. Instead, 'batch_size={}' was set ".format(
                    batch_size, len(X_cand)))
            batch_size = len(X_cand)

        # Check if the argument return_utilities is valid
        if not isinstance(return_utilities, bool):
            raise TypeError(
                '{} is an invalid type for return_utilities. Type {} is '
                'expected'.format(type(return_utilities), bool))

        # Check self.clf and self.method
        if not isinstance(self.method, str):
            raise TypeError("{} is an invalid type for the attribute "
                            "'self.method'.".format(type(self.method)))
        if self.method not in ['KL_divergence', 'vote_entropy']:
            raise ValueError(
                "The given method {} is not valid. Supported methods are "
                "'KL_divergence' and 'vote_entropy'".format(self.method))
        if self.method == 'vote_entropy' and \
                ((getattr(self._clf, 'fit', None) is None or
                  getattr(self._clf, 'predict', None) is None)):
            raise TypeError(
                "'clf' must implement the methods 'fit' and 'predict'")
        elif self.method == 'KL_divergence' and \
                ((getattr(self._clf, 'fit', None) is None or
                  getattr(self._clf, 'predict_proba', None) is None)):
            raise TypeError(
                "'clf' must implement the methods 'fit' and 'predict_proba'")

        # check self.ensemble and self.clf
        if not isinstance(self._clf, BaseEnsemble):
            if self.ensemble is None:
                warnings.warn('\'ensemble\' is not specified, '
                              '\'BaggingClassifier\' will be used.')
                ensemble = BaggingClassifier
            elif not callable(self.ensemble):
                raise TypeError("{} is not valid for the attribute "
                                "'self.ensemble'.".format(self.ensemble))
            elif not isinstance(self.ensemble(), BaseEnsemble):
                raise TypeError("{} is an invalid type for the attribute "
                                "'self.ensemble'.".format(type(self.ensemble)))
            else:
                ensemble = self.ensemble
            parameters = ensemble.__init__.__code__.co_varnames
            kwargs = self.kwargs_ensemble
            if 'base_estimator' in parameters:
                kwargs['base_estimator'] = self._clf
            self._clf = ensemble(random_state=random_state, **kwargs)

        if not isinstance(self._clf, SklearnClassifier):
            self._clf = SklearnClassifier(self._clf, classes=self.classes)

        self._clf.fit(X, y)
        # Check if the given classes are the same
        if self.classes is None:
            self.classes = self._clf.classes_
        if not np.array_equal(self._clf.classes_, self.classes):
            raise ValueError("The given classes are not the same as in the "
                             "classifier.")

        # choose the disagreement method and calculate the utilities
        if hasattr(self._clf, 'estimators_'):
            est_arr = self._clf.estimators_
        else:
            est_arr = [self._clf] * self._clf.n_estimators
        if self.method == 'KL_divergence':
            P = [est.predict_proba(X_cand) for est in est_arr]
            utilities = average_kl_divergence(P)
        elif self.method == 'vote_entropy':
            votes = [est.predict(X_cand) for est in est_arr]
            utilities = vote_entropy(votes, self.classes)

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def average_kl_divergence(probas):
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the
    level of disagreement in QBC.

    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_X_cand, n_classes)
        The probability estimations of all estimators, instances and classes.

    Returns
    -------
    scores: np.ndarray, shape (n_X_cand)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning
    for text classification. In Proceedings of the International Conference on
    Machine Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """

    # validation:
    # check P
    probas = check_array(probas, accept_sparse=False,
                         accept_large_sparse=True, dtype="numeric", order=None,
                         copy=False, force_all_finite=True, ensure_2d=True,
                         allow_nd=True, ensure_min_samples=1,
                         ensure_min_features=1, estimator=None)
    if probas.ndim != 3:
        raise ValueError("Expected 2D array, got 1D array instead:"
                         "\narray={}.".format(probas))

    # calculate the average KL divergence:
    probas = np.array(probas)
    probas_mean = np.mean(probas, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.nansum(
            np.nansum(probas * np.log(probas / probas_mean), axis=2), axis=0)
    scores = scores / probas.shape[0]

    return scores


def vote_entropy(votes, classes):
    """
    Calculate the vote entropy for measuring the level of disagreement in QBC.

    Parameters
    ----------
    votes : array-like, shape (n_estimators, n_X_cand)
        The class predicted by the estimators for each instance.
    classes : array-like, shape (n_classes)
        A list of all possible classes.

    Returns
    -------
    vote_entropy : np.ndarray, shape (n_X_cand))
        The vote entropy of each instance in X_cand.

    References
    ----------
    [1] Engelson, Sean P., and Ido Dagan.
    "Minimizing manual annotation cost in supervised training from corpora."
    arXiv preprint cmp-lg/9606030 (1996).
    """
    # check votes to be valid
    votes = check_array(votes, accept_sparse=False,
                        accept_large_sparse=True, dtype="numeric", order=None,
                        copy=False, force_all_finite=True, ensure_2d=True,
                        allow_nd=False, ensure_min_samples=1,
                        ensure_min_features=1, estimator=None)
    # Check classes to be valid
    check_classes(classes)

    # count the votes
    vote_count = np.zeros((votes.shape[1], len(classes)))
    for i in range(votes.shape[1]):
        for c_idx, c in enumerate(classes):
            for m in range(len(votes)):
                vote_count[i, c_idx] += (votes[m, i] == c)

    # compute vote entropy
    v = vote_count / len(votes)
    return -np.nansum(v*np.log(v), axis=1) / np.log(len(votes))
