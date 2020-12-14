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
    check_classes, ExtLabelEncoder, check_classifier_params
from ..classifier import SklearnClassifier


class UncertaintySampling(SingleAnnotPoolBasedQueryStrategy):
    """
    Uncertainty Sampling query strategy.

    Parameters
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    method : string (default='margin_sampling')
        The method to calculate the uncertainty, entropy, least_confident,
        margin_sampling, and expected_average_precision  are possible.
        Epistemic only works with Parzen Window Classifier or
        Logistic Regression.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i. Only supported for least confident
        variant.
    random_state : numeric | np.random.RandomState
        The random state to use.

    Attributes
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    method : string
        The method to calculate the uncertainty. Only entropy, least_confident,
        margin_sampling and expected_average_precisionare.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i. Only supported for least confident
        variant.
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
    """

    def __init__(self, clf, method='margin_sampling', cost_matrix=None,
                 random_state=None):
        super().__init__(random_state=random_state)

        self.method = method
        self.cost_matrix = cost_matrix
        self.clf = clf

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
            Sample weights.
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
        self._clf = clone(self.clf)

        # Check if the attribute clf is valid
        if not isinstance(self._clf, SkactivemlClassifier):
            raise TypeError('clf as to be from type SkactivemlClassifier. The #'
                            'given type is {}. Use the wrapper in '
                            'skactiveml.classifier to use a sklearn '
                            'classifier/ensemble.'.format(type(self._clf)))

        # check X, y and X_cand
        X, y, X_cand = check_X_y(X, y, X_cand, force_all_finite=False)

        # check random state
        random_state = check_random_state(self.random_state, len(X_cand))

        # Extract classes from clf
        label_encoder = ExtLabelEncoder(missing_label=self._clf.missing_label,
                                        classes=self.clf.classes).fit(y)
        classes = label_encoder.classes_

        # Check if the classifier and its arguments are valid
        check_classifier_params(classes, self._clf.missing_label)

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

        # check self.method
        if not isinstance(self.method, str):
            raise TypeError('{} is an invalid type for method. Type {} is '
                            'expected'.format(type(self.method), str))

        if self.method not in ['entropy', 'least_confident', 'margin_sampling',
                               'expected_average_precision']:
            raise ValueError(
                "The given method {} is not valid. Supported methods are "
                "'KL_divergence' and 'vote_entropy'".format(self.method))

        if getattr(self.clf, 'predict_proba', None) is None:
            raise TypeError("'clf' must implement the method 'predict_proba'")

        # fit the classifier and get the probabilities
        self._clf.fit(X, y, sample_weight=sample_weight)
        probas = self._clf.predict_proba(X_cand)

        # choose the method and calculate the utilities
        with np.errstate(divide='ignore'):
            if self.method in ['least_confident', 'margin_sampling',
                               'entropy']:
                utilities = uncertainty_scores(probas=probas,
                                               method=self.method,
                                               cost_matrix=self.cost_matrix)
            elif self.method == 'expected_average_precision':
                utilities = expected_average_precision(classes, probas)

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def uncertainty_scores(probas, cost_matrix=None, method='least_confident'):
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
    probas = check_array(probas, accept_sparse=False,
            accept_large_sparse=True, dtype="numeric", order=None,
            copy=False, force_all_finite=True, ensure_2d=True,
            allow_nd=False, ensure_min_samples=1,
            ensure_min_features=1, estimator=None)

    if (np.sum(probas, axis=1) - 1).all():
        raise ValueError('probas are invalid. The sum over axis 1 must be '
                         'one.')

    n_classes = probas.shape[1]

    # Check cost matrix.
    if cost_matrix is not None:
        cost_matrix = check_cost_matrix(cost_matrix, n_classes=n_classes)

    # Compute uncertainties.
    if method == 'least_confident':
        if cost_matrix is None:
            return 1 - np.max(probas, axis=1)
        else:
            costs = probas @ cost_matrix
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return costs[:, 0]
    elif method == 'margin_sampling':
        if cost_matrix is None:
            probas = -(np.partition(-probas, 1, axis=1)[:, :2])
            return 1 - np.abs(probas[:, 0] - probas[:, 1])
        else:
            costs = probas @ cost_matrix
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return -np.abs(costs[:, 0] - costs[:, 1])
    elif method == 'entropy':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nansum(-probas * np.log(probas), axis=1)
    else:
        raise ValueError(
            "Supported methods are ['least_confident', 'margin_sampling', "
            "'entropy'], the given one is: {}.".format(method)
        )


# expected average precision:
def expected_average_precision(classes, probas):
    """
    Calculate the expected average precision.

    Parameters
    ----------
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    probas : np.ndarray, shape=(n_X_cand, n_classes)
        The probabiliti estimation for each classes and all instance in X_cand.

    Returns
    -------
    score : np.ndarray, shape=(n_X_cand)
        The expected average precision score of all instances in X_cand.
    """
    # check if probas is valid
    probas = check_array(probas, accept_sparse=False,
            accept_large_sparse=True, dtype="numeric", order=None,
            copy=False, force_all_finite=True, ensure_2d=True,
            allow_nd=False, ensure_min_samples=1,
            ensure_min_features=1, estimator=None)

    if (np.sum(probas, axis=1) - 1).all():
        raise ValueError('probas are invalid. The sum over axis 1 must be '
                         'one.')

    # check if classes is valid
    check_classes(classes)
    if len(classes) < 2:
        raise ValueError('classes must contain at least 2 entries.')
    if len(classes) != probas.shape[1]:
        raise ValueError('classes must have the same length as probas has '
                         'columns.')

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
    return p[n - 1] * f_arr[n - 1, t - 1] + p[n - 1] * t * g_arr[n - 1, t - 1] / n + (1 - p[n - 1]) * f_arr[n - 1, t]

