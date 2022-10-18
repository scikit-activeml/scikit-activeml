import itertools

import numpy as np
from scipy.special import factorial, gammaln
from sklearn import clone
from sklearn.utils.validation import check_array

from ._expected_error_reduction import ExpectedErrorReduction
from .utils import IndexClassifierWrapper

from ..base import ClassFrequencyEstimator
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
    """(Multi-class) Probabilistic Active Learning

    This class implements multi-class probabilistic active learning (McPAL) [1]
    strategy.

    Parameters
    ----------
    prior: float, optional (default=1)
        Prior probabilities for the Dirichlet distribution of the samples.
    m_max: int, optional (default=1)
        Maximum number of hypothetically acquired labels.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra
        Spiliopoulou. Multi-Class Probabilistic Active Learning,
        vol. 285 of Frontiers in Artificial Intelligence and Applications,
        pages 586-594. IOS Press, 2016
    """

    def __init__(
        self, prior=1, m_max=1, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
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
        """Query the next instance to be labeled.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        clf : skactiveml.base.ClassFrequencyEstimator
            Model implementing the methods `fit` and `predict_freq`.
        fit_clf : bool, default=True
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        utility_weight: array-like, optional (default=None)
            Weight for each candidate (multiplied with utilities). Usually,
            this is to be the density of a candidate in ProbabilisticAL. The
            length of `utility_weight` is usually n_samples, except for the
            case when candidates contains samples (ndim >= 2). Then the length
            is `n_candidates`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
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
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Check the classifier's type.
        check_type(clf, "clf", ClassFrequencyEstimator)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_type(fit_clf, "fit_clf", bool)

        # Check `utility_weight`.
        if utility_weight is None:
            if mapping is None:
                utility_weight = np.ones(len(X_cand))
            else:
                utility_weight = np.ones(len(X))
        utility_weight = check_array(utility_weight, ensure_2d=False)

        if mapping is None and not len(X_cand) == len(utility_weight):
            raise ValueError(
                f"'utility_weight' must have length 'n_candidates' but "
                f"{len(X_cand)} != {len(utility_weight)}."
            )
        if mapping is not None and not len(X) == len(utility_weight):
            raise ValueError(
                f"'utility_weight' must have length 'n_samples' but "
                f"{len(X)} != {len(utility_weight)}."
            )

        # Fit the classifier and predict frequencies.
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)
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


class XPal(ExpectedErrorReduction):
    # TODO description, parameters
    """


    Parameters
    ----------
    method : string, default='inductive'

    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost matrix with cost_matrix[i,j] defining the cost of predicting class
        j for a sample with the actual class i.
    candidate_prior : float, default=1.e-3

    evaluation_prior : float, default=1.e-3

    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int | np.random.RandomState, default=None
        Random state for candidate selection.

    References
    ----------
    [1] Kottke, Daniel, et al. "Toward optimal probabilistic active learning
        using a Bayesian approach." Machine Learning 110.6 (2021): 1199-1231.
    """

    def __init__(self, method='inductive', cost_matrix=None,
                 candidate_prior=1.e-3, evaluation_prior=1.e-3,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(
            enforce_mapping=False,
            cost_matrix=cost_matrix,
            missing_label=missing_label,
            random_state=random_state,
        )
        self.method = method
        self.candidate_prior = candidate_prior
        self.evaluation_prior = evaluation_prior

    def query(self, X, y, clf, fit_clf=True, ignore_partial_fit=True,
              sample_weight=None,
              candidates=None, sample_weight_candidates=None,
              X_eval=None, sample_weight_eval=None,
              batch_size=1, return_utilities=False,
              return_candidate_utilities=False):
        # TODO describe parameter 'return_candidate_utilities'
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        ignore_partial_fit : bool, optional (default=True)
            Relevant in cases where `clf` implements `partial_fit`. If True,
            the `partial_fit` function is ignored and `fit` is used instead.
        sample_weight : array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        sample_weight_candidates : array-like of shape (n_candidates),
            optional (default=None)
            Weights of candidates samples in `candidates` if candidates are
            directly given (i.e., candidates.ndim > 1). Otherwise weights for
            candidates are given in `sample_weight`.
        X_eval : array-like of shape (n_eval_samples, n_features),
            optional (default=None).
            Unlabeled evalaution data set that is used for estimating the risk.
        sample_weight_eval : array-like of shape (n_eval_samples),
            optional (default=None)
            Weights of evaluation samples in `X_eval` if given. Used to weight
            the importance of samples when estimating the risk.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.
        return_candidate_utilities : bool, default=False


        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.candidate_prior
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        (
            X, y, sample_weight, clf, candidates, sample_weight_candidates,
            X_eval, sample_weight_eval, batch_size, return_utilities
        ) = self._validate_data(
            X, y, sample_weight, clf, candidates, sample_weight_candidates,
            X_eval, sample_weight_eval, batch_size, return_utilities,
            reset=True, check_X_dict=None
        )

        self._validate_init_params()

        if self.method == 'transductive':
            self.enforce_mapping = True
            if X_eval is not None:
                raise ValueError('X_eval must be None in the transductive '
                                 'active learning setting.')

        X_cand, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=self.enforce_mapping
        )

        X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval = \
            self._concatenate_samples(X, y, sample_weight,
                                      candidates, sample_weight_candidates,
                                      X_eval, sample_weight_eval)

        if self.method == 'transductive':
            idx_eval = idx_train

        # Check fit_clf
        check_type(fit_clf, 'fit_clf', bool)

        # Initialize classifier that works with indices to improve readability
        id_clf = IndexClassifierWrapper(
            clf, X_full, y_full, w_full, set_base_clf=not fit_clf,
            ignore_partial_fit=ignore_partial_fit, enforce_unique_samples=True,
            use_speed_up=True, missing_label=self.missing_label_
        )

        # Fit the classifier
        id_clf = self._precompute_and_fit_clf(id_clf, X_full, y_full,
                                              idx_train, idx_cand, idx_eval,
                                              fit_clf=fit_clf)

        # Fit the ground truth for candiates
        gt_clf_cand = ParzenWindowClassifier(
            classes=id_clf.classes_,
            metric=id_clf.clf.metric,
            metric_dict=id_clf.clf.metric_dict,
            class_prior=self.candidate_prior
        )
        gt_cand = IndexClassifierWrapper(
            gt_clf_cand, X_full, y_full, w_full, set_base_clf=False,
            ignore_partial_fit=ignore_partial_fit, use_speed_up=False,
            missing_label=self.missing_label_
        )

        # Compute class-membership probabilities of candidate samples
        gt_cand.fit(idx_train)
        probs_cand = gt_cand.predict_proba(idx_cand)

        # Fit the ground truth for evaluation instances
        gt_clf = ParzenWindowClassifier(
            classes=id_clf.classes_,
            metric=id_clf.clf.metric,
            metric_dict=id_clf.clf.metric_dict,
            class_prior=self.evaluation_prior
        )
        self.gt_eval = IndexClassifierWrapper(
            gt_clf, X_full, y_full, w_full, set_base_clf=False,
            ignore_partial_fit=ignore_partial_fit, use_speed_up=False,
            missing_label=self.missing_label_
        )
        self.gt_eval.precompute(idx_train, idx_eval, fit_params='labeled')
        self.gt_eval.precompute(idx_cand, idx_eval)
        self.gt_eval.fit(idx_train, set_base_clf=True)

        # Check cost matrix.
        classes = id_clf.classes_
        self._validate_cost_matrix(len(classes))

        # precomputating values before the loop
        self._precompute_loop(id_clf, idx_train, idx_cand, idx_eval, w_eval)

        # determine cgains
        if self.method == 'transductive':
            cost_est = self.cost_matrix_[:, self.pred_old_[idx_cand]].T
            self.cgain_ = np.sum(
                np.sum(
                    w_eval[idx_cand][:, np.newaxis] * probs_cand *
                    cost_est[np.newaxis, :],
                    axis=2
                ), axis=0
            )
        else:
            self.cgain_ = np.zeros(len(idx_cand))

        # Storage for computed errors per candidate sample
        errors = np.zeros([len(X_cand), len(classes)])

        # Iterate over candidate samples
        for i_cx, idx_cx in enumerate(idx_cand):
            # Simulate acquisition of label for each candidate sample and class
            for i_cy, cy in enumerate(classes):
                errors[i_cx, i_cy] = self._estimate_error_for_candidate(
                    id_clf, [idx_cx], [cy], idx_train, idx_cand, idx_eval,
                    w_eval
                )

        utilities_cand = (-np.sum(probs_cand * errors, axis=1) + self.cgain_)

        if mapping is None:
            utilities = utilities_cand
            cand_utilities = self.cgain_.reshape(1, 1)
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand
            cand_utilities = np.full([1, len(X)], np.nan)
            cand_utilities[0, mapping] = self.cgain_

        idx, utils = simple_batch(utilities, self.random_state_,
                                  batch_size=batch_size,
                                  return_utilities=True)

        if return_utilities is False and return_candidate_utilities is False:
            return idx
        elif return_utilities is True and return_candidate_utilities is False:
            return idx, utils
        elif return_utilities is False and return_candidate_utilities is True:
            return idx, cand_utilities
        else:
            return idx, utils, cand_utilities

    def _validate_init_params(self):
        methods = ['transducve', 'inductive']
        if not isinstance(self.method, str):
            raise TypeError('"method" has to be of type "str"')
        if self.method not in methods:
            raise ValueError(f'"method" has to be one of: {methods}.')

    def _precompute_and_fit_clf(self, id_clf, X_full, y_full,
                                idx_train, idx_cand, idx_eval, fit_clf):
        id_clf.precompute(idx_train, idx_cand)
        id_clf.precompute(idx_train, idx_eval)
        id_clf.precompute(idx_cand, idx_eval)
        id_clf = super()._precompute_and_fit_clf(
            id_clf, X_full, y_full, idx_train, idx_cand, idx_eval,
            fit_clf=fit_clf
        )

        return id_clf

    # TODO remove unused arguments
    def _precompute_loop(self, id_clf, idx_train, idx_cand, idx_eval, w_eval):
        self.pred_old_ = np.full(max(idx_eval) + 1, -1)
        self.pred_old_[idx_eval] = id_clf._le.transform(
            id_clf.predict(idx_eval))

    def _estimate_error_for_candidate(self, uclf, idx_cx, cy, idx_train,
                                      idx_cand, idx_eval, w_eval):

        if self.method == 'transductive':
            idx_eval_ = np.setdiff1d(idx_eval, idx_cx)
            if len(idx_eval_) == 0:
                return 0
        else:  # inductive
            idx_eval_ = idx_eval

        self.gt_eval.partial_fit(idx_cx, cy, use_base_clf=True,
                                 set_base_clf=False)
        probs_new = self.gt_eval.predict_proba(idx_eval_)

        uclf.partial_fit(idx_cx, cy, use_base_clf=True, set_base_clf=False)

        pred_new = uclf._le.transform(uclf.predict(idx_eval_))

        err_old = self._risk_estimation(probs_new, self.pred_old_[idx_eval_],
                                        self.cost_matrix_, w_eval[idx_eval_])

        err_new = self._risk_estimation(probs_new, pred_new,
                                        self.cost_matrix_, w_eval[idx_eval_])

        if self.method == 'transductive':
            return err_new - err_old
        else:
            return (err_new - err_old) / len(idx_eval_)