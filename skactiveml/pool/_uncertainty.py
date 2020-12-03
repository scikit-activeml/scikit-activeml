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

from ..base import SingleAnnotPoolBasedQueryStrategy, ClassFrequencyEstimator
from ..utils import rand_argmax, is_labeled, MISSING_LABEL, check_X_y, \
    check_scalar, check_cost_matrix, simple_batch, check_random_state
from ..classifier import SklearnClassifier


class UncertaintySampling(SingleAnnotPoolBasedQueryStrategy):
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
    method : string
        The method to calculate the uncertainty. Only entropy, least_confident,
        margin_sampling, expected_average_precisionare and epistemic.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    precompute : boolean (default=False)
        Whether the epistemic uncertainty should be precomputed.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.

    References
    ---------
    [1] Settles, Burr. Active learning literature survey.
        University of Wisconsin-Madison Department of Computer Sciences, 2009.
        http://www.burrsettles.com/pub/settles.activelearning.pdf

    [2] Wang, Hanmo, et al. "Uncertainty sampling for action recognition
        via maximizing expected average precision."
        IJCAI International Joint Conference on Artificial Intelligence. 2018.

    [3] Nguyen, Vu-Linh, Sébastien Destercke, and Eyke Hüllermeier.
        "Epistemic uncertainty sampling." International Conference on
        Discovery Science. Springer, Cham, 2019.
    """

    def __init__(self, clf, classes=None, method='margin_sampling',
                 precompute=False, missing_label=MISSING_LABEL,
                 random_state=None):
        super().__init__(random_state=random_state)

        self.missing_label = missing_label
        self.method = method
        self.classes = classes
        self.clf = clf
        self.is_precompute = precompute
        self.precompute_array = None

    def query(self, X_cand, X, y, batch_size=1, return_utilities=False, **kwargs):
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

        # validation:
        # check random state
        random_state = check_random_state(self.random_state)

        # check self.method
        if (self.method != 'entropy' and self.method != 'least_confident' and
                self.method != 'margin_sampling' and
                self.method != 'expected_average_precision' and
                self.method != 'epistemic'):
            warnings.warn("The method '" + self.method + "' does not exist,"
                                                         ",'margin_sampling' will be used.")
            self.method = 'margin_sampling'

        # checks for method=margin_sampling
        if (self.method == 'margin_sampling' and
                getattr(self.clf, 'predict_proba', None) is None):
            raise TypeError("'clf' must implement the method 'predict_proba'")

        # checks for method=expected_average_precision
        if (self.method == 'expected_average_precision' and
                self.classes is None):
            raise ValueError('\'classes\' has to be specified')

        # checks for method=epistemic
        if self.method == 'epistemic':
            if isinstance(self.clf, ClassFrequencyEstimator):
                self.method = 'epistemic_pwc'
            elif isinstance(self.clf, LogisticRegression):
                self.method = 'epistemic_logreg'
            else:
                raise TypeError("'clf' must be a subclass of"
                                "ClassFrequencyEstimator or"
                                "LogisticRegression")

        # check X, y and X_cand
        X, y, X_cand = check_X_y(X, y, X_cand, force_all_finite=False)

        # create precompute_array if necessary
        if self.is_precompute and self.precompute_array is None:
            self.precompute_array = np.full((2, 2), np.nan)

        # fit the classifier and get the probabilities
        mask_labeled = is_labeled(y, self.missing_label)
        self.clf = clone(self.clf)
        self.clf.fit(X, y)
        probas = self.clf.predict_proba(X_cand)

        # choose the method and calculate the utilities
        with np.errstate(divide='ignore'):
            if self.method in ['least_confident', 'margin_sampling',
                               'entropy']:
                utilities = uncertainty_scores(P=probas, method=self.method)
            elif self.method == 'expected_average_precision':
                utilities = expected_average_precision(
                    X_cand, self.classes, probas)
            elif self.method == 'epistemic_pwc':
                utilities, self.precompute_array = epistemic_uncertainty_pwc(
                    self.clf, X_cand, self.precompute_array)
            elif self.method == 'epistemic_logreg':
                return NotImplemented
                utilities = epistemic_uncertainty_logreg(
                    X[mask_labeled], y[mask_labeled], self.clf, probas)

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def uncertainty_scores(P, cost_matrix=None, method='least_confident'):
    """Computes uncertainty scores. Three methods are available: least
    confident ('least_confident'), margin sampling ('margin_sampling'),
    and entropy based uncertainty ('entropy') [1]. For the least confident and
    margin sampling methods cost-sensitive variants are implemented in case of
    a given cost matrix (see [2] for more information).

    Parameters
    ----------
    P : array-like, shape (n_samples, n_classes)
        Class membership probabilities for each sample.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i. Only supported for least confident
        variant.
    method : {'lc', 'sm', 'entropy'}, optional (default='lc')
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
    [2] Margineantu, Dragos D. "Active cost-sensitive learning."
        In IJCAI, vol. 5, pp. 1622-1623. 2005.
    """
    # Check probabilities.
    P = check_array(P)
    n_classes = P.shape[1]

    # Check cost matrix.
    if cost_matrix is not None:
        cost_matrix = check_cost_matrix(cost_matrix, n_classes=n_classes)

    # Compute uncertainties.
    if method == 'least_confident':
        if cost_matrix is None:
            return 1 - np.max(P, axis=1)
        else:
            costs = P @ cost_matrix
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return costs[:, 0]
    elif method == 'margin_sampling':
        if cost_matrix is None:
            P = -(np.partition(-P, 1, axis=1)[:, :2])
            return 1 - np.abs(P[:, 0] - P[:, 1])
        else:
            costs = P @ cost_matrix
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return -np.abs(costs[:, 0] - costs[:, 1])
    elif method == 'entropy':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nansum(-P * np.log(P), axis=1)
    else:
        raise ValueError(
            "Supported methods are ['least_confident', 'margin_sampling', "
            "'entropy'], the given one is: {}.".format(method)
        )


# expected average precision:
def expected_average_precision(X_cand, classes, probas):
    """
    Calculate the expected average precision.

    Parameters
    ----------
    X_cand : np.ndarray
        The unlabeled pool for which to calculated the expected average
        precision.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    probas : np.ndarray, shape=(n_X_cand, n_classes)
        The probabiliti estimation for each classes and all instance in X_cand.

    Returns
    -------
    score : np.ndarray, shape=(n_X_cand)
        The expected average precision score of all instances in X_cand.
    """
    score = np.zeros(len(X_cand))
    for i in range(len(classes)):
        for j, x in enumerate(X_cand):
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
    return p[n - 1] * f_arr[n - 1, t - 1] + p[n - 1] * t * g_arr[n - 1, t - 1] / n + (1 - p[n - 1]) * f_arr[n - 1, t]


# epistemic uncertainty:
def epistemic_uncertainty_pwc(clf, X_cand, precompute_array):
    freq = clf.predict_freq(X_cand)
    n = freq[:, 0]
    p = freq[:, 1]
    res = np.full((len(freq)), np.nan)
    if precompute_array is not None:
        # enlarges the precompute_array array if necessary:
        if precompute_array.shape[0] <= np.max(n) + 1:
            new_shape = (int(np.max(n)) - precompute_array.shape[0] + 2, precompute_array.shape[1])
            precompute_array = np.append(precompute_array, np.full(new_shape, np.nan), axis=0)
        if precompute_array.shape[1] <= np.max(p) + 1:
            new_shape = (precompute_array.shape[0], int(np.max(p)) - precompute_array.shape[1] + 2)
            precompute_array = np.append(precompute_array, np.full(new_shape, np.nan), axis=1)

        for f in freq:
            # compute the epistemic uncertainty:
            for N in range(precompute_array.shape[0]):
                for P in range(precompute_array.shape[1]):
                    if np.isnan(precompute_array[N, P]):
                        pi1 = -minimize_scalar(_epistemic_pwc_sup_1, method='Bounded', bounds=(0.0, 1.0),
                                               args=(N, P)).fun
                        pi0 = -minimize_scalar(_epistemic_pwc_sup_0, method='Bounded', bounds=(0.0, 1.0),
                                               args=(N, P)).fun
                        pi = np.array([pi0, pi1])
                        precompute_array[N, P] = np.min(pi, axis=0)
        res = _interpolate(precompute_array, freq)
    else:
        for i, f in enumerate(freq):
            pi1 = -minimize_scalar(_epistemic_pwc_sup_1, method='Bounded', bounds=(0.0, 1.0), args=(f[0], f[1])).fun
            pi0 = -minimize_scalar(_epistemic_pwc_sup_0, method='Bounded', bounds=(0.0, 1.0), args=(f[0], f[1])).fun
            pi = np.array([pi0, pi1])
            res[i] = np.min(pi, axis=0)
    return res, precompute_array


# bilinear interpolation for epistemic_uncertainty_pwc
def _interpolate(precompute_array, freq):
    points = np.zeros((precompute_array.shape[0] * precompute_array.shape[1], 2))
    for n in range(precompute_array.shape[0]):
        for p in range(precompute_array.shape[1]):
            points[n * precompute_array.shape[1] + p] = n, p
    return griddata(points, precompute_array.flatten(), freq, method='linear')


# support 1 for epistemic_uncertainty_pwc
def _epistemic_pwc_sup_1(t, n, p):
    if (n == 0.0) and (p == 0.0):
        return -1.0
    piH = ((t ** p) * ((1 - t) ** n)) / (((p / (n + p)) ** p) * ((n / (n + p)) ** n))
    return -np.minimum(piH, 2 * t - 1)


# support 1 for epistemic_uncertainty_pwc
def _epistemic_pwc_sup_0(t, n, p):
    if ((n == 0.0) and (p == 0.0)):
        return -1.0
    piH = ((t ** p) * ((1 - t) ** n)) / (((p / (n + p)) ** p) * ((n / (n + p)) ** n))
    return -np.minimum(piH, 1 - 2 * t)


# logistic regression epistemic_uncertainty_logreg
# alg 3
def epistemic_uncertainty_logreg(X_cand, X, y, clf, probas):
    # calculate the maximum likelihood of the logistic function
    theta_ml = np.insert(clf.coef_, len(X), clf.intercept_, axis=0)
    L_ml = np.exp(loglike_logreg(theta_ml, X, y, gamma=1))
    #
    x0 = np.zeros((X_cand.shape[1]+1))  #
    # compute pi0, pi1 for every x in X_cand:
    pi0, pi1 = np.empty((len(probas))), np.empty((len(probas)))
    for i, x in enumerate(X_cand):
        Qn = np.linspace(0.0, 0.5, num=50, endpoint=False)
        Qp = np.linspace(0.5, 1.0, num=50, endpoint=False)
        pi1[i], pi0[i] = np.maximum(2 * probas[i] - 1, 0), np.maximum(1 - 2 * probas[i], 0)
        #
        A = np.insert(x, len(x), 1)
        for q in range(100):
            idx_an, idx_ap = np.argmin(Qn), np.argmax(Qp)
            alpha_n, alpha_p = Qn[idx_an], Qp[idx_ap]
            if 2 * alpha_p - 1 > pi1[i]:
                # solve 22 -> theta
                bounds = np.log(alpha_p / (1 - alpha_p))
                constraints = LinearConstraint(A=A, lb=bounds, ub=bounds)
                theta = minimize(loglike_logreg, x0=x0, method='SLSQP', constraints=constraints, args=(X, y)).x  #
                pi1[i] = np.maximum(pi1[i], np.min(pi_h(theta, L_ml, X, y), 2 * alpha_p - 1))
            if 1 - 2 * alpha_n > pi0[i]:
                # solve 22 -> theta
                bounds = np.log(alpha_p / (1 - alpha_p))
                constraints = LinearConstraint(A=A, lb=bounds, ub=bounds)
                theta = minimize(loglike_logreg, x0=x0, method='SLSQP', constraints=constraints, args=(X, y)).x  #
                pi0[i] = np.maximum(pi0[i], np.min(pi_h(theta, L_ml, X, y), 1 - 2 * alpha_n))

            Qn, Qp = np.delete(Qn, idx_an), np.delete(Qp, idx_ap)

    utilities = np.min(np.array([pi0, pi1]), axis=1)
    return utilities


def loglike_logreg(theta, X, y, gamma=1):
    return -_logistic_loss(theta, X, y, gamma, sample_weight=None)


def pi_h(theta, L_ml, X, y, gamma=1):
    L_theta = np.exp(loglike_logreg(theta, X, y, gamma))
    return L_theta / L_ml
