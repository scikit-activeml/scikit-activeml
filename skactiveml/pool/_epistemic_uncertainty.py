"""
Uncertainty query strategies
"""

# Author: Pascal Mergard <Pascal.Mergard@student.uni-kassel.de>

import numpy as np
import warnings

from scipy.optimize import minimize_scalar, minimize, LinearConstraint
from scipy.interpolate import griddata

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_loss

from ..base import SingleAnnotPoolBasedQueryStrategy, ClassFrequencyEstimator, \
    SkactivemlClassifier
from ..utils import rand_argmax, is_labeled, MISSING_LABEL, check_X_y, \
    check_scalar, check_cost_matrix, simple_batch, check_random_state, \
    check_classes, ExtLabelEncoder
from ..classifier import SklearnClassifier, PWC


class EpistemicUncertainty(SingleAnnotPoolBasedQueryStrategy):
    """
    Uncertainty Sampling query strategy.

    Parameters
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    classes : array-like, shape=(n_classes), (default=None)
        Holds the label for each class.
    method : string (default='margin_sampling')
        The method to calculate the uncertainty, entropy, least_confident,
        margin_sampling, expected_average_precision and epistemic are possible.
        Epistemic only works with Parzen Window Classifier or
        Logistic Regression.
    precompute : boolean (default=False)
        Whether the epistemic uncertainty should be precomputed.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        The random state to use.

    Attributes
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    precompute : boolean (default=False)
        Whether the epistemic uncertainty should be precomputed.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.

    References
    ---------
    [1] Nguyen, Vu-Linh, Sébastien Destercke, and Eyke Hüllermeier.
        "Epistemic uncertainty sampling." International Conference on
        Discovery Science. Springer, Cham, 2019.
    """

    def __init__(self, clf, precompute=False, random_state=None):
        super().__init__(random_state=random_state)

        self.clf = clf
        self.precompute = precompute

    def query(self, X_cand, X, y, sample_weight=None, batch_size=1,
              return_utilities=False):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : np.ndarray
            The unlabeled pool from which to choose.
        X : np.ndarray
            The labeled pool used to fit the classifier.
        y : np.array
            The labels of the labeled pool X.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool (default=False)
            If True, the utilities are returned.

        Returns
        -------
        best_indices : np.ndarray, shape (batch_size)
            The index of the queried instance.
        batch_utilities : np.ndarray,  shape (batch_size, len(X_cand))
            The utilities of all instances of
            X_cand(if return_utilities=True).
        """
        # Validate input parameters.
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        X, y, sample_weight = check_X_y(X, y, sample_weight)

        # Check if the attribute clf is valid
        if not isinstance(self.clf, SkactivemlClassifier):
            raise TypeError(
                'clf as to be from type SkactivemlClassifier. The #'
                'given type is {}. Use the wrapper in '
                'skactiveml.classifier to use a sklearn '
                'classifier/ensemble.'.format(type(self.clf)))

        # fit the classifier and get the probabilities
        clf = clone(self.clf)
        clf.fit(X, y, sample_weight=sample_weight)

        # checks for method=epistemic
        # TODO sklearn.neighbors.RadiusNeighborsClassifier ???
        if isinstance(clf, PWC):
            if not hasattr(self, 'precompute_array'):
                self._precompute_array = None

            # create precompute_array if necessary
            if not isinstance(self.precompute, bool):
                raise TypeError(
                    '{} is an invalid type for precompute. Type {} is '
                    'expected'.format(type(self.precompute), bool))
            if self.precompute and self._precompute_array is None:
                self._precompute_array = np.full((2, 2), np.nan)

            freq = clf.predict_freq(X_cand)
            utilities, self._precompute_array = epistemic_uncertainty_pwc(
                freq, self._precompute_array)
        elif isinstance(clf, SklearnClassifier) and \
                isinstance(clf.estimator, LogisticRegression):
            mask_labeled = is_labeled(y, clf.missing_label)
            probas = clf.predict_proba(X_cand)
            utilities = epistemic_uncertainty_logreg(X_cand=X_cand,
                                                     X=X[mask_labeled],
                                                     y=y[mask_labeled],
                                                     clf=clf, probas=probas)
        else:
            raise TypeError("'clf' must be from type PWC or "
                            "a wrapped LogisticRegression classifier. "
                            "The given is from type {}."
                            "".format(type(self.clf)))

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


# epistemic uncertainty scores:
def epistemic_uncertainty_pwc(freq, precompute_array):
    if freq.shape[1] != 2:
        raise ValueError('epistemic is only implemented for two-class '
                         'problems, {} classes were given.'
                         ''.format(freq.shape[1]))
    n = freq[:, 0]
    p = freq[:, 1]
    res = np.full((len(freq)), np.nan)
    if precompute_array is not None:
        # enlarges the precompute_array array if necessary:
        if precompute_array.shape[0] < np.max(n) + 1:
            new_shape = (int(np.max(n)) - precompute_array.shape[0] + 2,
                         precompute_array.shape[1])
            old_shape = precompute_array.shape
            precompute_array = np.append(precompute_array,
                                         np.full(new_shape, np.nan), axis=0)
        if precompute_array.shape[1] < np.max(p) + 1:
            new_shape = (precompute_array.shape[0],
                         int(np.max(p)) - precompute_array.shape[1] + 2)
            precompute_array = np.append(precompute_array,
                                         np.full(new_shape, np.nan), axis=1)

        # precompute the epistemic uncertainty:
        for N in range(precompute_array.shape[0]):
            for P in range(precompute_array.shape[1]):
                if np.isnan(precompute_array[N, P]):
                    pi1 = -minimize_scalar(
                        _epistemic_pwc_sup_1, method='Bounded',
                        bounds=(0.0, 1.0), args=(N, P)
                    ).fun

                    pi0 = -minimize_scalar(
                        _epistemic_pwc_sup_0, method='Bounded',
                        bounds=(0.0, 1.0), args=(N, P)
                    ).fun

                    pi = np.array([pi0, pi1])
                    precompute_array[N, P] = np.min(pi, axis=0)
        res = _interpolate(precompute_array, freq)
    else:
        for i, f in enumerate(freq):
            pi1 = -minimize_scalar(
                _epistemic_pwc_sup_1, method='Bounded', bounds=(0.0, 1.0),
                args=(f[0], f[1])).fun

            pi0 = -minimize_scalar(
                _epistemic_pwc_sup_0, method='Bounded', bounds=(0.0, 1.0),
                args=(f[0], f[1])).fun

            pi = np.array([pi0, pi1])
            res[i] = np.min(pi, axis=0)
    return res, precompute_array


# bilinear interpolation for epistemic_uncertainty_pwc
def _interpolate(precompute_array, freq):
    points = np.zeros(
        (precompute_array.shape[0] * precompute_array.shape[1], 2)
    )
    for n in range(precompute_array.shape[0]):
        for p in range(precompute_array.shape[1]):
            points[n * precompute_array.shape[1] + p] = n, p
    return griddata(points, precompute_array.flatten(), freq, method='linear')


# support 1 for epistemic_uncertainty_pwc
def _epistemic_pwc_sup_1(theta, n, p):
    if (n == 0.0) and (p == 0.0):
        return -1.0
    piH = ((theta ** p) * ((1 - theta) ** n)) / \
          (((p / (n + p)) ** p) * ((n / (n + p)) ** n))
    return -np.minimum(piH, 2 * theta - 1)


# support 0 for epistemic_uncertainty_pwc
def _epistemic_pwc_sup_0(theta, n, p):
    if ((n == 0.0) and (p == 0.0)):
        return -1.0
    piH = ((theta ** p) * ((1 - theta) ** n)) / \
          (((p / (n + p)) ** p) * ((n / (n + p)) ** n))
    return -np.minimum(piH, 1 - 2 * theta)


# logistic regression epistemic_uncertainty_logreg
# alg 3
def epistemic_uncertainty_logreg(X_cand, X, y, clf, probas):
    if len(clf.classes) != 2:
        raise ValueError('epistemic is only implemented for two-class '
                         'problems, {} classes were given.'
                         ''.format(len(clf.classes)))
    # calculate the maximum likelihood of the logistic function
    theta_ml = np.append(clf.coef_, clf.intercept_).flatten()
    L_ml = np.exp(-_loglike_logreg(theta_ml, X, y, gamma=1))
    #
    x0 = np.zeros((X_cand.shape[1] + 1))  #
    # compute pi0, pi1 for every x in X_cand:
    pi1 = np.maximum(2 * probas[:, 0] - 1, 0)
    pi0 = np.maximum(1 - 2 * probas[:, 0], 0)
    for i, x in enumerate(X_cand):
        Qn = np.linspace(0.01, 0.5, num=50, endpoint=True)
        Qp = np.linspace(0.5, 1.0, num=50, endpoint=False)
        #
        A = np.append(x, 1)  # used for the LinearConstraint
        for q in range(50):
            alpha_n, alpha_p = Qn[0], Qp[-1]
            if 2 * alpha_p - 1 > pi1[i]:
                # solve 22 -> theta
                bounds = np.log(alpha_p / (1 - alpha_p))
                constraints = LinearConstraint(A=A, lb=bounds, ub=bounds)
                theta = minimize(_loglike_logreg, x0=x0, method='SLSQP',
                                 constraints=constraints, args=(X, y)).x  #
                pi1[i] = np.maximum(
                    pi1[i], np.minimum(_pi_h(theta=theta, L_ml=L_ml, X=X, y=y),
                                       2 * alpha_p - 1)
                )
            if 1 - 2 * alpha_n > pi0[i]:
                # solve 22 -> theta
                bounds = np.log(alpha_n / (1 - alpha_n))
                constraints = LinearConstraint(A=A, lb=bounds, ub=bounds)
                theta = minimize(_loglike_logreg, x0=x0, method='SLSQP',
                                 constraints=constraints, args=(X, y)).x  #
                pi0[i] = np.maximum(
                    pi0[i], np.minimum(_pi_h(theta=theta, L_ml=L_ml, X=X, y=y),
                                       1 - 2 * alpha_n)
                )
            Qn, Qp = np.delete(Qn, 0), np.delete(Qp, -1)

    utilities = np.min(np.array([pi0, pi1]), axis=0)
    print(utilities)
    return utilities


def _pi_h(theta, L_ml, X, y, gamma=1):
    L_theta = np.exp(-_loglike_logreg(theta, X, y, gamma))
    return L_theta / L_ml


def _loglike_logreg(theta, X, y, gamma=1):
    # TODO sample_weights ???
    return _logistic_loss(theta, X, y, gamma, sample_weight=None)
