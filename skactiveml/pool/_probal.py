import itertools

import numpy as np
from scipy.special import factorial, gammaln
from sklearn.utils.validation import check_array

from ..base import ClassFrequencyEstimator
from ..base import SingleAnnotPoolBasedQueryStrategy
from ..utils import check_scalar, simple_batch, check_type, fit_if_not_fitted


class McPAL(SingleAnnotPoolBasedQueryStrategy):
    """Multi-class Probabilistic Active Learning

    This class implements multi-class probabilistic active learning (McPAL) [1]
    strategy.

    Parameters
    ----------
    prior: float, optional (default=1)
        Prior probabilities for the Dirichlet distribution of the samples.
    m_max: int, optional (default=1)
        Maximum number of hypothetically acquired labels.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra
        Spiliopoulou. Multi-Class Probabilistic Active Learning,
        vol. 285 of Frontiers in Artificial Intelligence and Applications,
        pages 586-594. IOS Press, 2016
    """

    def __init__(self, prior=1, m_max=1, random_state=None):
        super().__init__(random_state=random_state)
        self.prior = prior
        self.m_max = m_max

    def query(self, X_cand, clf, X=None, y=None, sample_weight=None,
              utility_weight=None, batch_size=1, return_utilities=False):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand : array-like, shape (n_candidate_samples, n_features)
            Candidate samples from which the strategy can select.
        clf : skactiveml.base.ClassFrequencyEstimator
            Model implementing the methods `fit` and `predict_proba`.
        X: array-like, shape (n_samples, n_features), optional (default=None)
            Complete training data set.
        y: array-like, shape (n_samples), optional (default=None)
            Labels of the training data set.
        sample_weight: array-like, shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        utility_weight: array-like (n_candidate_samples), optional
        (default=None)
            Densities for each sample in `X`.
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
        X_cand, return_utilities, batch_size, utility_weight, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                utility_weight, self.random_state, reset=True)

        # Check the classifier's type.
        check_type(clf, ClassFrequencyEstimator, 'clf')

        # Fit the classifier and predict frequencies.
        clf = fit_if_not_fitted(clf, X, y, sample_weight)
        k_vec = clf.predict_freq(X_cand)

        # Calculate utilities and return the output.
        utilities = utility_weight * cost_reduction(k_vec, prior=self.prior,
                                                    m_max=self.m_max)

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)

    def _validate_data(self, X_cand, return_utilities, batch_size,
                       utility_weight, random_state, reset=True):
        X_cand, return_utilities, batch_size, random_state = \
            super()._validate_data(X_cand, return_utilities, batch_size,
                                   self.random_state, reset=True)

        # Check `utility_weight`.
        if utility_weight is None:
            utility_weight = np.ones(len(X_cand))
        utility_weight = check_array(utility_weight, ensure_2d=False)

        # Check if `X_cand` and `utility_weight` have the same length.
        if not len(X_cand) == len(utility_weight):
            raise ValueError(
                "'X_cand' and 'utility_weight' must have the same length."
            )

        return X_cand, return_utilities, batch_size, utility_weight, \
            random_state


def cost_reduction(k_vec_list, C=None, m_max=2, prior=1.e-3):
    """Calculate the expected cost reduction.

    Calculate the expected cost reduction for given maximum number of
    hypothetically acquired labels, observed labels and cost matrix.

    Parameters
    ----------
    k_vec_list: array-like, shape (n_samples, n_classes)
        Observed class labels.
    C: array-like, shape = (n_classes, n_classes)
        Cost matrix.
    m_max: int
        Maximal number of hypothetically acquired labels.
    prior : float | array-like, shape (n_classes)
       Prior value for each class.

    Returns
    -------
    expected_cost_reduction: array-like, shape (n_samples)
        Expected cost reduction for given parameters.
    """
    # Check if 'prior' is valid
    check_scalar(prior, 'prior', (float, int),
                 min_inclusive=False, min_val=0)

    # Check if 'm_max' is valid
    check_scalar(m_max, 'm_max', int, min_val=1)

    n_classes = len(k_vec_list[0])
    n_samples = len(k_vec_list)

    # check cost matrix
    C = 1 - np.eye(n_classes) if C is None else np.asarray(C)

    # generate labelling vectors for all possible m values
    l_vec_list = np.vstack([_gen_l_vec_list(m, n_classes)
                            for m in range(m_max + 1)])
    m_list = np.sum(l_vec_list, axis=1)
    n_l_vecs = len(l_vec_list)

    # compute optimal cost-sensitive decision for all combination of k-vectors
    # and l-vectors
    tile = np.tile(k_vec_list, (n_l_vecs, 1, 1))
    k_l_vec_list = np.swapaxes(tile, 0, 1) + l_vec_list
    y_hats = np.argmin(k_l_vec_list @ C, axis=2)

    # add prior to k-vectors
    prior = prior * np.ones(n_classes)
    k_vec_list = np.asarray(k_vec_list) + prior

    # all combination of k-, l-, and prediction indicator vectors
    combs = [k_vec_list, l_vec_list, np.eye(n_classes)]
    combs = np.asarray([list(elem)
                        for elem in list(itertools.product(*combs))])

    # three factors of the closed form solution
    factor_1 = 1 / _euler_beta(k_vec_list)
    factor_2 = _multinomial(l_vec_list)
    factor_3 = _euler_beta(np.sum(combs, axis=1)).reshape(n_samples, n_l_vecs,
                                                          n_classes)

    # expected classification cost for each m
    m_sums = np.asarray(
        [factor_1[k_idx]
         * np.bincount(m_list, factor_2 * [C[:, y_hats[k_idx, l_idx]]
                                           @ factor_3[k_idx, l_idx]
                                           for l_idx in range(n_l_vecs)])
         for k_idx in range(n_samples)]
    )

    # compute classification cost reduction as difference
    gains = np.zeros((n_samples, m_max)) + m_sums[:, 0].reshape(-1, 1)
    gains -= m_sums[:, 1:]

    # normalize  cost reduction by number of hypothetical label acquisitions
    gains /= np.arange(1, m_max + 1)

    return np.max(gains, axis=1)


def _gen_l_vec_list(m_approx, n_classes):
    """
    Creates all possible class labeling vectors for given number of
    hypothetically acquired labels and given number of classes.

    Parameters
    ----------
    m_approx: int
        Number of hypothetically acquired labels..
    n_classes: int,
        Number of classes

    Returns
    -------
    label_vec_list: array-like, shape = [n_labelings, n_classes]
        All possible class labelings for given parameters.
    """

    label_vec_list = [[]]
    label_vec_res = np.arange(m_approx + 1)
    for i in range(n_classes - 1):
        new_label_vec_list = []
        for labelVec in label_vec_list:
            for newLabel in label_vec_res[label_vec_res
                                          - (m_approx - sum(labelVec))
                                          <= 1.e-10]:
                new_label_vec_list.append(labelVec + [newLabel])
        label_vec_list = new_label_vec_list

    new_label_vec_list = []
    for labelVec in label_vec_list:
        new_label_vec_list.append(labelVec + [m_approx - sum(labelVec)])
    label_vec_list = np.array(new_label_vec_list, int)

    return label_vec_list


def _euler_beta(a):
    """
    Represents Euler beta function:
    B(a(i)) = Gamma(a(i,1))*...*Gamma(a_n)/Gamma(a(i,1)+...+a(i,n))

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Euler beta function results [B(a(0)), ..., B(a(m))
    """
    return np.exp(np.sum(gammaln(a), axis=1) - gammaln(np.sum(a, axis=1)))


def _multinomial(a):
    """
    Computes Multinomial coefficient:
    Mult(a(i)) = (a(i,1)+...+a(i,n))!/(a(i,1)!...a(i,n)!)

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Multinomial coefficients [Mult(a(0)), ..., Mult(a(m))
    """
    return factorial(np.sum(a, axis=1)) / np.prod(factorial(a), axis=1)
