import itertools

import numpy as np
from scipy.special import factorial, gammaln
from sklearn import clone
from sklearn.utils.validation import check_array

from ..base import SkactivemlClassifier
from ..base import SingleAnnotatorPoolQueryStrategy
from ..classifier import ParzenWindowClassifier
from ..utils import (
    MISSING_LABEL,
    check_scalar,
    simple_batch,
    check_type,
    check_equal_missing_label,
)


class ProbabilisticAL(SingleAnnotatorPoolQueryStrategy):
    """Multi-class Probabilistic Active Learning (McPAL)

    This class implements the query strategy Multi-class Probabilistic Active
    Learning (McPAL) [1]_, which estimates the performance gain when labeling
    samples.

    Parameters
    ----------
    prior : float, default=1.0
        Prior probabilities for the Dirichlet distribution of the samples.
    m_max : int, default=1.0
        Maximum number of hypothetically acquired labels.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    metric : str or callable, default=None
        The metric must be None or a valid kernel as defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`. The kernel is used to
        calculate the frequency of labels near the candidates and multiplied
        with the probabilities returned by `clf` to get a kernel frequency
        estimate for each class.
        If metric is set to None, the `predict_freq` function of the `clf` will
        be used instead. If this is not defined, a TypeError is raised.
    metric_dict : dict, default=None
        Any further parameters that should be passed directly to the kernel
        function. If metric_dict is None and metric is 'rbf' metric_dict is set
        to {'gamma': 'mean'}.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] D. Kottke, G. Krempl, D. Lang, J. Teschner, and M. Spiliopoulou.
       Multi-class Probabilistic Active Learning. In Eur. Conf. Artif. Intell.,
       pages 586–594, 2016.
    """

    def __init__(
        self,
        prior=1,
        m_max=1,
        missing_label=MISSING_LABEL,
        metric=None,
        metric_dict=None,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.metric = metric
        self.metric_dict = metric_dict
        self.prior = prior
        self.m_max = m_max

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        utility_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Query the next samples to be labeled.

        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`.)
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight: array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        utility_weight : array-like, default=None
            Weight for each candidate (multiplied with utilities). Usually,
            this is to be the density of a candidate in ProbabilisticAL. The
            length of `utility_weight` is usually n_samples, except for the
            case when candidates contains samples (ndim >= 2). Then the length
            is `n_candidates`.
        candidates : None or array-like of shape (n_candidates, ) of type \
                int, default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, `candidates` is
              considered as the candidate samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is to
            be queried, e.g., `query_indices[0]` indicates the first selected
            sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None`, the indexing refers to the samples
              in `X`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `utilities` refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, *)`, `utilities`
              refers to the indexing in `candidates`.
        """
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Check the classifier's type.
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_type(fit_clf, "fit_clf", bool)

        # Check `utility_weight`.
        if utility_weight is None:
            if mapping is None:
                utility_weight = np.ones(len(X_cand))
            else:
                utility_weight = np.ones(len(X))
        utility_weight = check_array(utility_weight, ensure_2d=False)

        if mapping is None and len(X_cand) != len(utility_weight):
            raise ValueError(
                f"'utility_weight' must have length 'n_candidates' but "
                f"{len(X_cand)} != {len(utility_weight)}."
            )
        if mapping is not None and len(X) != len(utility_weight):
            raise ValueError(
                f"'utility_weight' must have length 'n_samples' but "
                f"{len(X)} != {len(utility_weight)}."
            )

        if self.metric is None and not hasattr(clf, "predict_freq"):
            raise TypeError(
                "clf has no predict_freq and metric was set to None"
            )

        # Fit the classifier and predict frequencies.
        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)
        if self.metric is not None:
            if self.metric_dict is None and self.metric == "rbf":
                self.metric_dict = {"gamma": "mean"}
            pwc = ParzenWindowClassifier(
                metric=self.metric,
                metric_dict=self.metric_dict,
                missing_label=clf.missing_label,
                classes=clf.classes,
            )
            pwc.fit(X=X, y=y, sample_weight=sample_weight)
            n = pwc.predict_freq(X_cand).sum(axis=1, keepdims=True)
            pred_proba = clf.predict_proba(X_cand)
            k_vec = n * pred_proba
        else:
            k_vec = clf.predict_freq(X_cand)

        # Calculate utilities and return the output.
        utilities_cand = cost_reduction(
            k_vec, prior=self.prior, m_max=self.m_max
        )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand
        utilities *= utility_weight

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def cost_reduction(k_vec_list, C=None, m_max=2, prior=1.0e-3):
    """Calculate the expected cost reduction.

    Calculate the expected cost reduction for given maximum number of
    hypothetically acquired labels, observed labels and cost matrix.

    Parameters
    ----------
    k_vec_list : array-like of shape (n_samples, n_classes)
        Observed class labels.
    C : array-like of shape (n_classes, n_classes), default=None
        Cost matrix.
    m_max : int
        Maximal number of hypothetically acquired labels.
    prior : float or array-like of shape (n_classes,)
       Prior value for each class.

    Returns
    -------
    expected_cost_reduction: array-like, shape (n_samples)
        Expected cost reduction for given parameters.
    """
    # Check if 'prior' is valid
    check_scalar(prior, "prior", (float, int), min_inclusive=False, min_val=0)

    # Check if 'm_max' is valid
    check_scalar(m_max, "m_max", int, min_val=1)

    n_classes = len(k_vec_list[0])
    n_samples = len(k_vec_list)

    # check cost matrix
    C = 1 - np.eye(n_classes) if C is None else np.asarray(C)

    # generate labelling vectors for all possible m values
    l_vec_list = np.vstack(
        [_gen_l_vec_list(m, n_classes) for m in range(m_max + 1)]
    )
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
    combs = np.asarray(
        [list(elem) for elem in list(itertools.product(*combs))]
    )

    # three factors of the closed form solution
    factor_1 = 1 / _euler_beta(k_vec_list)
    factor_2 = _multinomial(l_vec_list)
    factor_3 = _euler_beta(np.sum(combs, axis=1)).reshape(
        n_samples, n_l_vecs, n_classes
    )

    # expected classification cost for each m
    m_sums = np.asarray(
        [
            factor_1[k_idx]
            * np.bincount(
                m_list,
                factor_2
                * [
                    C[:, y_hats[k_idx, l_idx]] @ factor_3[k_idx, l_idx]
                    for l_idx in range(n_l_vecs)
                ],
            )
            for k_idx in range(n_samples)
        ]
    )

    # compute classification cost reduction as difference
    gains = np.zeros((n_samples, m_max)) + m_sums[:, 0].reshape(-1, 1)
    gains -= m_sums[:, 1:]

    # normalize  cost reduction by number of hypothetical label acquisitions
    gains /= np.arange(1, m_max + 1)

    return np.max(gains, axis=1)


def _gen_l_vec_list(m_approx, n_classes):
    """Creates all possible class labeling vectors for given number of
    hypothetically acquired labels and given number of classes.

    Parameters
    ----------
    m_approx : int
        Number of hypothetically acquired labels.
    n_classes : int
        Number of classes.

    Returns
    -------
    label_vec_list : array-like of shape (n_labels, n_classes)
        All possible class label lists for given parameters.
    """

    label_vec_list = [[]]
    label_vec_res = np.arange(m_approx + 1)
    for i in range(n_classes - 1):
        new_label_vec_list = []
        for labelVec in label_vec_list:
            for newLabel in label_vec_res[
                label_vec_res - (m_approx - sum(labelVec)) <= 1.0e-10
            ]:
                new_label_vec_list.append(labelVec + [newLabel])
        label_vec_list = new_label_vec_list

    new_label_vec_list = []
    for labelVec in label_vec_list:
        new_label_vec_list.append(labelVec + [m_approx - sum(labelVec)])
    label_vec_list = np.array(new_label_vec_list, int)

    return label_vec_list


def _euler_beta(a):
    """Represents Euler beta function:

    B(a(i)) = (Gamma(a(i,1))*...*Gamma(a_n))/Gamma(a(i,1)+...+a(i,n)).

    Parameters
    ----------
    a : array-like of shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result : array-like of shape (m,)
        Euler beta function results [B(a(0)), ..., B(a(m))
    """
    return np.exp(np.sum(gammaln(a), axis=1) - gammaln(np.sum(a, axis=1)))


def _multinomial(a):
    """
    Computes Multinomial coefficient:

    Mult(a(i)) = (a(i,1)+...+a(i,n))!/(a(i,1)!...a(i,n)!)

    Parameters
    ----------
    a : array-like of shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result : array-like of shape (m)
        Multinomial coefficients [Mult(a(0)), ..., Mult(a(m)].
    """
    return factorial(np.sum(a, axis=1)) / np.prod(factorial(a), axis=1)
