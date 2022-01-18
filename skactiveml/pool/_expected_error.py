from copy import deepcopy

import numpy as np
from sklearn.base import clone
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from ..base import SingleAnnotPoolBasedQueryStrategy, SkactivemlClassifier
from ..classifier import PWC
from ..utils import check_type, is_labeled, simple_batch, \
    fit_if_not_fitted, check_cost_matrix, check_X_y, MISSING_LABEL, \
    check_equal_missing_label, labeled_indices, unlabeled_indices


class ExpectedErrorReduction_old(SingleAnnotPoolBasedQueryStrategy):
    """Expected Error Reduction

    This class implements the expected error reduction algorithm with different
    loss functions:
     - log loss (log_loss) [1],
     - expected misclassification risk (emr) [2],
     - and cost-sensitive learning (csl) [3].

    Parameters
    ----------
    method: {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.
    cost_matrix: array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Only supported for least confident
        variant.
    ignore_partial_fit: bool, optional (default=False)
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for annotator selection.

    References
    ----------
    [1] Roy, N., & McCallum, A. (2001). Toward optimal active learning through
        monte carlo estimation of error reduction. ICML, (pp. 441-448).
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. P. (2012).
        Scalable active learning for multiclass image classification.
        IEEE TrPAMI, 34(11), pp. 2259-2273.
    [3] Margineantu, D. D. (2005). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """

    LOG_LOSS = 'log_loss' # [1] eval on X (ind. of X_cand)
    ZERO_ONE = 'zero_one' # [1] eval on X (ind. of X_cand)
    EMR = 'emr' # [2] eval on X without x_cand (mapping req.), vgl. Kapoor
                    # (diff: exclude_labeled_samples)
    CSL = 'csl' # [3] eval on labeled in X,y (ind. of X_cand)

    def __init__(self, method=EMR, cost_matrix=None, ignore_partial_fit=False,
                 random_state=None):
        super().__init__(random_state=random_state)
        self.method = method
        self.cost_matrix = cost_matrix
        self.ignore_partial_fit = ignore_partial_fit

    def query(self, X, y, clf, fit_clf=True, sample_weight=None,
              candidates=None, batch_size=1, return_utilities=False):
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
        fit_clf : bool, default=True
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
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
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

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
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
                X, y, candidates, batch_size, return_utilities, reset=True
            )

        X_cand, mapping = self._transform_candidates(candidates, X, y)
        sample_weight_cand = sample_weight[mapping]

        # Calculate utilities
        utilities = expected_error_reduction(
            clf=clf, X_cand=X_cand, X=X, y=y, cost_matrix=self.cost_matrix,
            method=self.method, sample_weight=sample_weight,
            sample_weight_cand=sample_weight_cand,
            ignore_partial_fit=self.ignore_partial_fit
        )

        return simple_batch(utilities, self.random_state_,
                            batch_size=batch_size,
                            return_utilities=return_utilities)

class ExpectedErrorReduction():

    def _predict_prob_new(self, ):
        pass

    def _risk_estimation(self, prob_true, prob_pred, cost_matrix,
                         sample_weight):
        if prob_true.ndim == 1 and prob_pred.ndim == 1:
            cost_est = cost_matrix[prob_true, :][len(prob_true), prob_pred]
            return np.sum(sample_weight * cost_est)
        elif prob_true.ndim == 1 and prob_pred.ndim == 2:
            cost_est = cost_matrix[prob_true, :]
            return np.sum(sample_weight[:, np.newaxis] *
                          prob_pred * cost_est[np.newaxis, :])
        elif prob_true.ndim == 2 and prob_pred.ndim == 1:
            cost_est = cost_matrix[:, prob_pred].T
            return np.sum(sample_weight[:, np.newaxis] *
                          prob_true * cost_est[np.newaxis, :])
        else:
            prob_mat = prob_true[:, :, np.newaxis]@prob_pred[:, np.newaxis, :]
            return np.sum(sample_weight[:, np.newaxis, np.newaxis] *
                          prob_mat * cost_matrix[np.newaxis, :, :])

    def _logloss_estimation(self, prob_true, prob_pred):
        return -np.sum(prob_true * np.log(prob_pred + np.finfo(float).eps))

    def _clf_refit(self, clf, X, y, w, cx, cy, cw, partial_fit):
        # Create sample array for the retraining of the classifier.
        X_new = np.vstack((X, [cx])) if not partial_fit else np.array([cx])
        # Create label array for the retraining of the classifier.
        y_new = np.append(y, [[cy]]) if not partial_fit else np.array([cy])
        # Create sample_weight for the retraining of the classifier.
        if w is not None and cw is not None:
            w_new = np.append(w, [[cw]]) if not partial_fit else np.array([cw])
            args = [w_new]
        elif w is None and cw is None:
            args = []
        else:
            raise ValueError('`sample_weight` must be either None or set '
                             'for all samples')

        # Check whether sample weights are used.
        if partial_fit:
            return deepcopy(clf).partial_fit(X_new, y_new, *args)
        else:
            return clone(clf).fit(X_new, y_new, *args)


class MonteCarloEER(SingleAnnotPoolBasedQueryStrategy, ExpectedErrorReduction):
    """
    Roy McCallum
    """
    def __init__(self, method='misclassification_loss', cost_matrix=None,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.method = method
        self.cost_matrix = cost_matrix

    def query(self, X, y, clf, sample_weight=None,
              fit_clf=True, ignore_partial_fit=True,
              candidates=None, sample_weight_candidates=None,
              X_eval=None, sample_weight_eval=None,
              batch_size=1, return_utilities=False):

        enforce_mapping = False
        reset = True

        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
                X, y, candidates, batch_size, return_utilities, reset=reset
            )

        X_cand, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=enforce_mapping
        )

        # Validate classifier type.
        check_type(clf, 'clf', SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        # Validate method.
        if not isinstance(self.method, str):
            raise TypeError('{} is an invalid type for method. Type {} is '
                            'expected'.format(type(self.method), str))

        # Fit the classifier.
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)
        else:
            check_is_fitted(clf)
            clf = deepcopy(clf)

        # determine existing classes
        classes = clf.classes_
        n_classes = len(classes)

        # Check cost matrix.
        cost_matrix = 1 - np.eye(n_classes) if self.cost_matrix is None \
            else self.cost_matrix
        cost_matrix = check_cost_matrix(cost_matrix, n_classes)

        # Check if candidates are samples if sample_weight_candidates is set
        if (candidates is None or candidates.ndim == 1) and \
                sample_weight_candidates is not None:
            raise ValueError('Attribute `sample_weight_candidates` can only'
                             'be None if `candidates` consists of samples.')

        # Check if X_eval is set if sample_weight_eval is set
        if X_eval is None and sample_weight_eval is not None:
            raise ValueError('If `X_eval` is None, `sample_weight_eval` must'
                             'also be None')

        # TODO: test sample weight_eval - length + column

        # use partial fit if applicable
        check_type(ignore_partial_fit, 'ignore_partial_fit', bool)
        partial_fit = hasattr(clf, 'partial_fit') and not ignore_partial_fit

        # METHOD SPECIFIC

        X_full = X
        y_full = y
        w_full = sample_weight
        idx_train = np.arange(len(X))
        idx_lbld = labeled_indices(y)
        idx_unld = unlabeled_indices(y)

        if candidates is None:
            idx_cand = idx_unld
        elif candidates.ndim == 1:
            idx_cand = candidates
        else:
            X_full = np.concatenate([X_full, candidates], axis=0)
            y_full = np.concatenate([y_full, np.full(len(candidates), np.nan)],
                                    axis=0)
            if not (w_full is None and sample_weight_candidates is None):
                w_full = np.concatenate([w_full, sample_weight_candidates],
                                        axis=0)
            idx_cand = np.arange(len(X), len(X_full))

        if X_eval is None:
            idx_eval = idx_train
            if sample_weight_eval is None:
                w_eval = np.ones(len(idx_eval))
            else:
                if len(sample_weight_eval) == len(X):
                    w_eval = sample_weight_eval
                else:
                    raise ValueError('If `sample_weight_eval` is set but '
                                     '`X_eval` is None, then it should have '
                                     'same size as `X`')
        else:
            X_full = np.concatenate([X_full, X_eval], axis=0)
            y_full = np.concatenate([y_full, np.full(len(X_eval), np.nan)],
                                    axis=0)
            idx_eval = np.arange(len(X_full) - len(X_eval), len(X_full))
            if sample_weight_eval is None:
                w_eval = np.ones(len(X_eval))

        # sample_weight is checked by clf when fitted

        # Compute class-membership probabilities of candidate samples.
        probs_cand = clf.predict_proba(X_cand)

        # precompute PWC
        if isinstance(clf, PWC):
            metric = clf.metric
            metric_dict = {} if clf.metric_dict is None else clf.metric_dict
            clf.metric = 'precomputed'
            clf.metric_dict = {}

            K = np.full([len(X_full), len(idx_eval)], np.nan)
            train_idx = np.concatenate([idx_lbld, idx_cand], axis=0)
            K[train_idx, :] = \
                pairwise_kernels(X_full[train_idx], X_full[idx_eval],
                                 metric, **metric_dict)

        # Storage for computed errors per candidate sample.
        errors = np.zeros([len(X_cand), n_classes])

        # Iterate over candidate samples
        for i_cx, idx_cx in enumerate(idx_cand):
            idx_add = np.concatenate([idx_train, [idx_cx]])
            X_add = X_full[idx_add]
            # Simulate acquisition of label for each candidate sample and class.
            for i_cy, cy in enumerate(classes):

                y_add = np.concatenate([y_full[idx_train], [cy]])
                if w_full is None:
                    w_add = None
                else:
                    w_add = w_full[idx_add]

                if partial_fit:
                    clf_new = deepcopy(clf).partial_fit(X_add, y_add, w_add)
                else:
                    clf_new = clf.fit(X_add, y_add, w_add) # TODO clone deletes classes

                ### clf_new, X_full, y_full, w_full, w_eval, K, idx1,2,3
                # self.cost_matrix_, self.method, self.

                if isinstance(clf, PWC):
                    probs = clf_new.predict_proba(K[idx_add, :].T)
                else:
                    probs = clf_new.predict_proba(X_full[idx_eval, :])

                if self.method == 'misclassification_loss':
                    preds = np.argmax(np.dot(probs, cost_matrix), axis=1)
                    err = self._risk_estimation(probs, preds, cost_matrix,
                                                sample_weight_eval)
                elif self.method == 'log_loss':
                    err = self._logloss_estimation(probs, probs)
                else:
                    raise ValueError(
                        f"Supported methods are `misclassification_loss`, or"
                        f"`log_loss` the given one is: {self.method}"
                    )
                ###

                errors[i_cx, i_cy] = err

        utilities_cand = np.sum(probs_cand * errors, axis=1)

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(utilities, self.random_state_,
                            batch_size=batch_size,
                            return_utilities=return_utilities)



def expected_error_reduction(clf, X_cand, X=None, y=None, cost_matrix=None,
                             method='emr', sample_weight_cand=None,
                             sample_weight=None, ignore_partial_fit=False):
    """Compute uncertainty scores.

    In case of a given cost matrix C, maximum expected cost is implemented as
    score.

    Parameters
    ----------
    clf : skactiveml.base.SkactivemlClassifier
        Model implementing the methods `fit` and `predict_proba`.
    X_cand : array-like, shape (n_candidate_samples, n_features)
        Candidate samples from which the strategy can select.
    X : array-like, shape (n_samples, n_features), optional (default=None)
        Complete training data set.
    y : array-like, shape (n_samples), optional (default=None)
        Labels of the training data set.
    cost_matrix : array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Only supported for least confident
        variant.
    method : {'log_loss', 'emr', 'csl'}, optional (default='emr')
        Variant of expected error reduction to be used: 'log_loss' is
        cost-insensitive, while 'emr' and 'csl' are cost-sensitive variants.
    sample_weight : array-like, shape (n_samples), optional
    (default=None)
        Weights of training samples in `X`.
    sample_weight_cand : array-like, shape (n_candidate_samples), optional
    (default=None)
        Weights of candidate samples in `X_cand`.
    ignore_partial_fit : bool, optional (default=False)
        If false, the classifier will be refitted through `partial_fit` if
        available. Otherwise, the use of `fit` is enforced.

    Returns
    -------
    utilities : np.ndarray, shape (n_candidates)
        The utilities of all unlabeled instances.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. (2009, June).
        Multi-class active learning for image classification.
        In 2009 IEEE Conference on Computer Vision and Pattern Recognition
        (pp. 2372-2379). IEEE.
    [3] Margineantu, D. D. (2005, July). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """
    # Check if the classifier and its arguments are valid.
    check_type(clf, 'clf', SkactivemlClassifier)

    # Check whether to use `fit` or `partial_fit`.
    check_type(ignore_partial_fit, 'ignore_partial_fit', bool)
    use_fit = ignore_partial_fit or not hasattr(clf, 'partial_fit')
    if use_fit and (X is None or y is None):
        raise ValueError(
            '`X` and `y` cannot be None for a classifier using `fit` for '
            'retraining.'
        )
    if (X is None or y is None) and method == 'csl':
        raise ValueError(
            "`X` and `y` cannot be None for `method='csl'`."
        )
    use_sample_weight = sample_weight is not None \
                        or sample_weight_cand is not None
    if use_fit and (bool(sample_weight is None)
                    != bool(sample_weight_cand is None)):
        raise ValueError(
            '`sample_weight` and `sample_weight_cand` must either both be '
            'None or array-like, if the fit method is used.'
        )
    X, y, X_cand, sample_weight, sample_weight_cand = check_X_y(
        X, y, X_cand, sample_weight, sample_weight_cand,
        force_all_finite=False, missing_label=clf.missing_label,
        allow_nd=True
    )

    # Refit classifier.
    if use_sample_weight:
        clf = fit_if_not_fitted(clf, X, y, sample_weight, False)
    else:
        clf = fit_if_not_fitted(clf, X, y, None, False)
    clf_refit = clone(clf).fit if use_fit else deepcopy(clf).partial_fit

    # Check cost matrix.
    n_classes = len(clf.classes_)
    cost_matrix = 1 - np.eye(len(clf.classes_)) if cost_matrix is None else \
        cost_matrix
    cost_matrix = check_cost_matrix(cost_matrix, n_classes)

    # Compute class-membership probabilities of candidate samples.
    P = clf.predict_proba(X_cand)

    # Storage for computed errors per candidate sample.
    errors = np.zeros(len(X_cand))
    errors_per_class = np.zeros(n_classes)

    # Iterate over candidate samples
    for i, x in enumerate(X_cand):
        # Simulate acquisition of label for each candidate sample and class.
        for yi in range(n_classes):
            # Create sample array for the retraining of the classifier.
            X_new = np.vstack((X, [x])) if use_fit else np.array([x])
            # Create label array for the retraining of the classifier.
            y_new = np.append(y, [[yi]]) if use_fit else np.array([yi])
            # Check whether sample weights are used.
            if use_sample_weight:
                # Create sample weight array for the retraining of the
                # classifier.
                w = sample_weight_cand[i]
                if use_fit:
                    sample_weight_new = np.append(sample_weight, [[w]])
                else:
                    sample_weight_new = np.array([w])
                # Retrain classifier with sample weights.
                clf_new = clf_refit(X_new, y_new, sample_weight_new)
            else:
                # Retrain classifier without sample weights.
                clf_new = clf_refit(X_new, y_new)
            if method == 'emr':
                P_new = clf_new.predict_proba(X_cand)
                costs = np.sum((P_new.T[:, None] * P_new.T).T * cost_matrix)
            elif method == 'csl':
                is_lbld = is_labeled(y, clf_new.missing_label)
                X_labeled = X[is_lbld]
                y_labeled = y[is_lbld]
                y_indices = [np.where(clf_new.classes_ == label)[0][0]
                             for label in y_labeled]
                if len(X_labeled) > 0:
                    costs = np.sum(
                        clf_new.predict_proba(X_labeled) *
                        cost_matrix[y_indices]
                    )
                else:
                    costs = 0
            elif method == 'log_loss':
                P_new = clf_new.predict_proba(X_cand)
                costs = -np.sum(P_new * np.log(P_new + np.finfo(float).eps))
            else:
                raise ValueError(
                    f"Supported methods are [{ExpectedErrorReduction.EMR}, "
                    f"{ExpectedErrorReduction.CSL}, "
                    f"{ExpectedErrorReduction.LOG_LOSS}], the given one is: "
                    f"{method}"
                )
            errors_per_class[yi] = P[i, yi] * costs
        errors[i] = errors_per_class.sum()
    return -errors
