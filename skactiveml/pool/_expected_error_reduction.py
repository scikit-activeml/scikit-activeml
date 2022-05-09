import numpy as np

from .utils import IndexClassifierWrapper
from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    check_type,
    is_labeled,
    simple_batch,
    check_cost_matrix,
    MISSING_LABEL,
    check_equal_missing_label,
    unlabeled_indices,
    is_unlabeled,
)


class ExpectedErrorReduction(SingleAnnotatorPoolQueryStrategy):
    """Abstract class for Expected Error Reduction (EER).

    This class implements the basic workflow of EER algorithms containing:
     - determining ever candidates x label pair and simulate its outcome
       in the classifier by simulating it
     - determining some kind of risk for the new classifier

    These structure has been used by e.g.:
     - Roy, N., & McCallum, A. (2001). Toward optimal active learning through
       monte carlo estimation of error reduction. ICML, pp. 441-448.
     - Kapoor, A., Horvitz, E., & Basu, S. (2007). Selective Supervision:
       Guiding Supervised Learning with Decision-Theoretic Active Learning.
       IJCAI, pp. 877-882.
     - Margineantu, D. D. (2005). Active cost-sensitive learning.
       IJCAI, pp. 1622-1623.
     - Joshi, A. J., Porikli, F., & Papanikolopoulos, N. P. (2012). Scalable
       active learning for multiclass image classification.
       IEEE TrPAMI, 34(11), pp. 2259-2273.

    Parameters
    ----------
    enforce_mapping : bool
        If True, an exception is raised when no exact mapping between
        instances in `X` and instances in `candidates` can be determined.
    cost_matrix: array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Used for misclassification loss and ignored for log loss.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] Roy, N., & McCallum, A. (2001). Toward optimal active learning through
        monte carlo estimation of error reduction. ICML, (pp. 441-448).
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. P. (2012).
        Scalable active learning for multiclass image classification.
        IEEE TrPAMI, 34(11), pp. 2259-2273.
    [3] Margineantu, D. D. (2005). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    [4] Kapoor, Ashish, Eric Horvitz, and Sumit Basu. "Selective Supervision:
        Guiding Supervised Learning with Decision-Theoretic Active Learning."
        IJCAI. Vol. 7. 2007.
    """

    def __init__(
        self,
        enforce_mapping,
        cost_matrix=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.cost_matrix = cost_matrix
        self.enforce_mapping = enforce_mapping

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        ignore_partial_fit=True,
        sample_weight=None,
        candidates=None,
        sample_weight_candidates=None,
        X_eval=None,
        sample_weight_eval=None,
        batch_size=1,
        return_utilities=False,
    ):
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
        sample_weight : array-like of shape (n_samples), optional
        (default=None)
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
            Unlabeled evaluation data set that is used for estimating the risk.
            Not applicable for all EER methods.
        sample_weight_eval : array-like of shape (n_eval_samples),
            optional (default=None)
            Weights of evaluation samples in `X_eval` if given. Used to weight
            the importance of samples when estimating the risk.
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
        (
            X,
            y,
            sample_weight,
            clf,
            candidates,
            sample_weight_candidates,
            X_eval,
            sample_weight_eval,
            batch_size,
            return_utilities,
        ) = self._validate_data(
            X,
            y,
            sample_weight,
            clf,
            candidates,
            sample_weight_candidates,
            X_eval,
            sample_weight_eval,
            batch_size,
            return_utilities,
            reset=True,
            check_X_dict=None,
        )

        _, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=self.enforce_mapping
        )

        (
            X_full,
            y_full,
            w_full,
            w_eval,
            idx_train,
            idx_cand,
            idx_eval,
        ) = self._concatenate_samples(
            X,
            y,
            sample_weight,
            candidates,
            sample_weight_candidates,
            X_eval,
            sample_weight_eval,
        )

        # Check fit_clf
        check_type(fit_clf, "fit_clf", bool)

        # Initialize classifier that works with indices to improve readability
        id_clf = IndexClassifierWrapper(
            clf,
            X_full,
            y_full,
            w_full,
            set_base_clf=not fit_clf,
            ignore_partial_fit=ignore_partial_fit,
            enforce_unique_samples=True,
            use_speed_up=True,
            missing_label=self.missing_label_,
        )

        # Fit the classifier.
        id_clf = self._precompute_and_fit_clf(
            id_clf,
            X_full,
            y_full,
            idx_train,
            idx_cand,
            idx_eval,
            fit_clf=fit_clf,
        )
        # Compute class-membership probabilities of candidate samples
        probs_cand = id_clf.predict_proba(idx_cand)

        # Check cost matrix.
        classes = id_clf.classes_
        self._validate_cost_matrix(len(classes))

        # precomputating current error
        current_error = self._estimate_current_error(
            id_clf, idx_train, idx_cand, idx_eval, w_eval
        )

        # Storage for computed errors per candidate sample
        errors = np.zeros([len(idx_cand), len(classes)])

        # Iterate over candidate samples
        for i_cx, idx_cx in enumerate(idx_cand):
            # Simulate acquisition of label for each candidate sample and class
            for i_cy, cy in enumerate(classes):
                errors[i_cx, i_cy] = self._estimate_error_for_candidate(
                    id_clf,
                    [idx_cx],
                    [cy],
                    idx_train,
                    idx_cand,
                    idx_eval,
                    w_eval,
                )

        # utils are maximized, errors minimized: hence multiply by (-1)
        future_error = np.sum(probs_cand * errors, axis=1)
        utilities_cand = -1 * (future_error - current_error)

        if mapping is None:
            utilities = np.array(utilities_cand)
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )

    def _validate_data(
        self,
        X,
        y,
        sample_weight,
        clf,
        candidates,
        sample_weight_candidates,
        X_eval,
        sample_weight_eval,
        batch_size,
        return_utilities,
        reset=True,
        check_X_dict=None,
    ):

        # Validate input parameters.
        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=reset,
            check_X_dict=check_X_dict,
        )

        # Validate classifier type.
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        self._validate_init_params()

        return (
            X,
            y,
            sample_weight,
            clf,
            candidates,
            sample_weight_candidates,
            X_eval,
            sample_weight_eval,
            batch_size,
            return_utilities,
        )

    def _validate_init_params(self):
        """Function used to evaluate parameters of the `__init__` function that
        are not part of the abstract class to avoid redundancies.
        """
        pass

    def _precompute_and_fit_clf(
        self,
        id_clf,
        X_full,
        y_full,
        idx_train,
        idx_cand,
        idx_eval,
        fit_clf=True,
    ):
        if fit_clf:
            id_clf.fit(idx_train, set_base_clf=True)
        return id_clf

    def _estimate_current_error(
        self, id_clf, idx_train, idx_cand, idx_eval, w_eval
    ):
        """
        Result must be of float or of shape (len(idx_eval))
        """
        return 0.0

    def _estimate_error_for_candidate(
        self, uclf, idx_cx, cy, idx_train, idx_cand, idx_eval, w_eval
    ):
        raise NotImplementedError(
            "Error estimation method must be implemented"
            "by the query strategy."
        )

    def _validate_cost_matrix(self, n_classes):

        cost_matrix = (
            1 - np.eye(n_classes)
            if self.cost_matrix is None
            else self.cost_matrix
        )
        self.cost_matrix_ = check_cost_matrix(cost_matrix, n_classes)

    def _concatenate_samples(
        self,
        X,
        y,
        sample_weight,
        candidates,
        sample_weight_candidates,
        X_eval,
        sample_weight_eval,
    ):

        # Check if candidates are samples if sample_weight_candidates is set
        if (
            candidates is None or candidates.ndim == 1
        ) and sample_weight_candidates is not None:
            raise ValueError(
                "Attribute `sample_weight_candidates` can only "
                "be set if `candidates` consists of samples."
            )

        # TODO: test sample weight_eval - length + column

        if sample_weight is not None and len(X) != len(sample_weight):
            raise ValueError(
                "If `sample_weight` is set, it must have same "
                "length as `X`."
            )

        if sample_weight_candidates is not None and len(candidates) != len(
            sample_weight_candidates
        ):
            raise ValueError(
                "If `sample_weight` is set, it must have same "
                "length as `X`."
            )

        # Concatenate samples
        X_full = X
        y_full = y
        w_full = sample_weight
        idx_train = np.arange(len(X))
        idx_unld = unlabeled_indices(y, self.missing_label_)

        if candidates is None:
            idx_cand = idx_unld
        elif candidates.ndim == 1:
            idx_cand = candidates
        else:
            X_full = np.concatenate([X_full, candidates], axis=0)
            y_full = np.concatenate(
                [y_full, np.full(len(candidates), np.nan)], axis=0
            )
            if not (w_full is None and sample_weight_candidates is None):
                if w_full is None:
                    w_full = np.ones(len(X))
                if sample_weight_candidates is None:
                    sample_weight_candidates = np.ones(len(candidates))
                w_full = np.concatenate(
                    [w_full, sample_weight_candidates], axis=0
                )
            idx_cand = np.arange(len(X), len(X_full))

        if X_eval is None:
            idx_eval = idx_train
            if sample_weight_eval is None:
                w_eval = np.ones(len(X_full))
            else:
                if len(sample_weight_eval) != len(idx_eval):
                    raise ValueError(
                        "If `sample_weight_eval` is set but "
                        "`X_eval` is None, then it should have "
                        "same size as `X`"
                    )
                w_eval = np.zeros(len(X_full))
                w_eval[idx_eval] = sample_weight_eval
        else:
            X_full = np.concatenate([X_full, X_eval], axis=0)
            y_full = np.concatenate(
                [y_full, np.full(len(X_eval), np.nan)], axis=0
            )
            idx_eval = np.arange(len(X_full) - len(X_eval), len(X_full))
            w_eval = np.ones(len(X_full))
            if sample_weight_eval is not None:
                if len(sample_weight_eval) != len(idx_eval):
                    raise ValueError(
                        "If `sample_weight_eval` and `X_eval` "
                        "are set, then `sample_weight_eval` "
                        "should have len(X_eval)"
                    )
                w_eval[idx_eval] = sample_weight_eval
            if w_full is not None:
                w_full = np.concatenate([w_full, sample_weight_eval], axis=0)

        return X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval

    def _risk_estimation(
        self, prob_true, prob_pred, cost_matrix, sample_weight
    ):
        if prob_true.ndim == 1 and prob_pred.ndim == 1:
            cost_est = cost_matrix[prob_true, :][
                range(len(prob_true)), prob_pred
            ]
            return np.sum(sample_weight * cost_est)
        elif prob_true.ndim == 1 and prob_pred.ndim == 2:
            cost_est = cost_matrix[prob_true, :]
            return np.sum(
                sample_weight[:, np.newaxis]
                * prob_pred
                * cost_est[np.newaxis, :]
            )
        elif prob_true.ndim == 2 and prob_pred.ndim == 1:
            cost_est = cost_matrix[:, prob_pred].T
            return np.sum(
                sample_weight[:, np.newaxis]
                * prob_true
                * cost_est[np.newaxis, :]
            )
        else:
            prob_mat = (
                prob_true[:, :, np.newaxis] @ prob_pred[:, np.newaxis, :]
            )
            return np.sum(
                sample_weight[:, np.newaxis, np.newaxis]
                * prob_mat
                * cost_matrix[np.newaxis, :, :]
            )

    def _logloss_estimation(self, prob_true, prob_pred):
        return -np.sum(prob_true * np.log(prob_pred + np.finfo(float).eps))


class MonteCarloEER(ExpectedErrorReduction):
    """This class implements the expected error method from [1] that uses a
    Monte-Carlo approach to estimate the error.

    Therefore, it implements the following two steps:
     - determining ever candidates x label pair and simulate its outcome
       in the classifier by simulating it
     - determining some kind of risk for the new classifier

    Parameters
    ----------
    method : string, optional (default='misclassification_loss')
        The optimization method. Possible values are 'misclassification_loss'
        and 'log_loss'.
    cost_matrix: array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Used for misclassification loss and ignored for log loss.
    subtract_current : bool, optional (default=False)
        If True, the current error estimate is subtracted from the simulated
        score. This might be helpful to define a stopping criterion.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] Roy, N., & McCallum, A. (2001). Toward optimal active learning through
        monte carlo estimation of error reduction. ICML, (pp. 441-448)."""

    def __init__(
        self,
        method="misclassification_loss",
        cost_matrix=None,
        subtract_current=False,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            enforce_mapping=False,
            cost_matrix=cost_matrix,
            missing_label=missing_label,
            random_state=random_state,
        )
        self.method = method
        self.subtract_current = subtract_current

    def _validate_init_params(self):
        super()._validate_init_params()
        # Validate method.
        if not isinstance(self.method, str):
            raise TypeError(
                "{} is an invalid type for method. Type {} is "
                "expected".format(type(self.method), str)
            )
        if self.method not in ["misclassification_loss", "log_loss"]:
            raise ValueError(
                f"Supported methods are `misclassification_loss`, or"
                f"`log_loss` the given one is: {self.method}"
            )

        check_type(self.subtract_current, "subtract_current", bool)

        if self.method == "log_loss" and self.cost_matrix is not None:
            raise ValueError(
                "`cost_matrix` must be None if `method` is set to `log_loss`"
            )

    def _estimate_current_error(
        self, id_clf, idx_train, idx_cand, idx_eval, w_eval
    ):
        if self.subtract_current:
            probs = id_clf.predict_proba(idx_eval)
            if self.method == "misclassification_loss":
                preds = np.argmin(np.dot(probs, self.cost_matrix_), axis=1)
                err = self._risk_estimation(
                    probs, preds, self.cost_matrix_, w_eval[idx_eval]
                )
            elif self.method == "log_loss":
                err = self._logloss_estimation(probs, probs)
            return err
        else:
            return super()._estimate_current_error(
                id_clf, idx_train, idx_cand, idx_eval, w_eval
            )

    def _estimate_error_for_candidate(
        self, id_clf, idx_cx, cy, idx_train, idx_cand, idx_eval, w_eval
    ):
        id_clf.partial_fit(idx_cx, cy, use_base_clf=True, set_base_clf=False)
        probs = id_clf.predict_proba(idx_eval)

        if self.method == "misclassification_loss":
            preds = np.argmin(np.dot(probs, self.cost_matrix_), axis=1)
            err = self._risk_estimation(
                probs, preds, self.cost_matrix_, w_eval[idx_eval]
            )
        elif self.method == "log_loss":
            err = self._logloss_estimation(probs, probs)
        return err

    def _precompute_and_fit_clf(
        self, id_clf, X_full, y_full, idx_train, idx_cand, idx_eval, fit_clf
    ):
        id_clf.precompute(idx_train, idx_cand)
        id_clf.precompute(idx_train, idx_eval)
        id_clf.precompute(idx_cand, idx_eval)
        id_clf = super()._precompute_and_fit_clf(
            id_clf,
            X_full,
            y_full,
            idx_train,
            idx_cand,
            idx_eval,
            fit_clf=fit_clf,
        )
        return id_clf


class ValueOfInformationEER(ExpectedErrorReduction):
    """This class implements the expected error method from [1] that estimates
    the value of information. This method can be extended in a way that it also
    implements [2] and [3]. The default parameters describe [1].

    Therefore, it implements the following two steps:
     - determining ever candidates x label pair and simulate its outcome
       in the classifier by simulating it
     - determining some kind of risk for the new classifier

    Parameters
    ----------
    cost_matrix: array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Used for misclassification loss and ignored for log loss.
    consider_unlabeled : bool, optional (default=True)
        If True, the error is estimated on the unlabeled samples.
    consider_labeled : bool, optional (default=True)
        If True, the error is estimated on the labeled samples.
    candidate_to_labeled : bool, optional (default=True)
        If True, the candidate with the simulated label is added to the labeled
        set. As this label is considered to be correct, it will be evaluated
        under the `consider_labeled` flag then.
    subtract_current : bool, optional (default=False)
        If True, the current error estimate is subtracted from the simulated
        score. This might be helpful to define a stopping criterion as in [2].
    normalize : bool, optional (default=False)
        If True the error terms are normalized by the number of evaluation
        samples such that the errors represent the average error instead of the
        summed error. This will be done independently for the simulated and the
        current error.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] Kapoor, Ashish, Eric Horvitz, and Sumit Basu. "Selective Supervision:
        Guiding Supervised Learning with Decision-Theoretic Active Learning."
        IJCAI. Vol. 7. 2007.
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. P. (2012).
        Scalable active learning for multiclass image classification.
        IEEE TrPAMI, 34(11), pp. 2259-2273.
    [3] Margineantu, D. D. (2005). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """

    def __init__(
        self,
        cost_matrix=None,
        consider_unlabeled=True,
        consider_labeled=True,
        candidate_to_labeled=True,
        subtract_current=False,
        normalize=False,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            enforce_mapping=True,
            cost_matrix=cost_matrix,
            missing_label=missing_label,
            random_state=random_state,
        )
        self.consider_unlabeled = consider_unlabeled
        self.consider_labeled = consider_labeled
        self.candidate_to_labeled = candidate_to_labeled
        self.subtract_current = subtract_current
        self.normalize = normalize

    def _validate_init_params(self):
        super()._validate_init_params()
        check_type(self.consider_unlabeled, "consider_unlabeled", bool)
        check_type(self.consider_labeled, "consider_labeled", bool)
        check_type(self.candidate_to_labeled, "candidate_to_labeled", bool)
        check_type(self.subtract_current, "subtract_current", bool)
        check_type(self.normalize, "normalize", bool)

    def query(
        self,
        X,
        y,
        clf,
        sample_weight=None,
        fit_clf=True,
        ignore_partial_fit=True,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
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
        sample_weight : array-like of shape (n_samples), optional
        (default=None)
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
        # TODO check if candidates are only unlabeled ones if given
        return super().query(
            X,
            y,
            clf,
            sample_weight=sample_weight,
            fit_clf=fit_clf,
            ignore_partial_fit=ignore_partial_fit,
            candidates=candidates,
            sample_weight_candidates=None,
            X_eval=None,
            sample_weight_eval=None,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )

    def _estimate_error_for_candidate(
        self, id_clf, idx_cx, cy, idx_train, idx_cand, idx_eval, w_eval
    ):
        id_clf.partial_fit(idx_cx, cy, use_base_clf=True, set_base_clf=False)

        # Handle problem that if only one candidate is remaining, this should
        # be the one to be selected although the error cannot be estimated
        # as there are no instances left for estimating

        le = id_clf._le
        y_eval = id_clf.y[idx_eval]
        idx_labeled = idx_train[is_labeled(y_eval)]
        y_labeled = id_clf.y[idx_labeled]
        idx_unlabeled = idx_train[is_unlabeled(y_eval)]

        if self.candidate_to_labeled:
            idx_labeled = np.concatenate([idx_labeled, idx_cx], axis=0)
            y_labeled = np.concatenate([y_labeled, cy], axis=0)
            idx_unlabeled = np.setdiff1d(
                idx_unlabeled, idx_cx, assume_unique=True
            )

        y_labeled_c_id = le.transform(y_labeled)

        err = 0
        norm = 0
        if self.consider_labeled and len(idx_labeled) > 0:
            norm += len(idx_labeled)
            probs = id_clf.predict_proba(idx_labeled)
            err += self._risk_estimation(
                y_labeled_c_id, probs, self.cost_matrix_, w_eval[idx_labeled]
            )

        if self.consider_unlabeled and len(idx_unlabeled) > 0:
            norm += len(idx_unlabeled)
            probs = id_clf.predict_proba(idx_unlabeled)
            err += self._risk_estimation(
                probs, probs, self.cost_matrix_, w_eval[idx_unlabeled]
            )

        if self.normalize:
            if norm == 0:
                return 0.0
            else:
                return err / norm
        else:
            return err

    def _estimate_current_error(
        self, id_clf, idx_train, idx_cand, idx_eval, w_eval
    ):
        # estimate current utility score if required
        # TODO: maybe use function for code below to reduce redundancies
        if self.subtract_current:
            le = id_clf._le
            y_eval = id_clf.y[idx_eval]
            idx_labeled = idx_train[is_labeled(y_eval)]
            y_labeled = id_clf.y[idx_labeled]
            idx_unlabeled = idx_train[is_unlabeled(y_eval)]

            y_labeled_c_id = le.transform(y_labeled)

            err = 0
            norm = 0
            if self.consider_labeled and len(idx_labeled) > 0:
                norm += len(idx_labeled)
                probs = id_clf.predict_proba(idx_labeled)
                err += self._risk_estimation(
                    y_labeled_c_id,
                    probs,
                    self.cost_matrix_,
                    w_eval[idx_labeled],
                )

            if self.consider_unlabeled and len(idx_unlabeled) > 0:
                norm += len(idx_unlabeled)
                probs = id_clf.predict_proba(idx_unlabeled)
                err += self._risk_estimation(
                    probs, probs, self.cost_matrix_, w_eval[idx_unlabeled]
                )

            if self.normalize:
                return err / norm
            else:
                return err
        else:
            return super()._estimate_current_error(
                id_clf, idx_train, idx_cand, idx_eval, w_eval
            )

    def _precompute_and_fit_clf(
        self, id_clf, X_full, y_full, idx_train, idx_cand, idx_eval, fit_clf
    ):

        # TODO: replace the following line by more efficient code
        id_clf.precompute(
            idx_train, idx_train, fit_params="all", pred_params="all"
        )

        #
        # # for cond_prob
        # id_clf.precompute(idx_train, idx_cand,
        #                   fit_params='labeled', pred_params='all')
        # # for risk estimation
        # if self.consider_labeled:
        #     id_clf.precompute(idx_train, idx_eval,
        #                       fit_params='labeled', pred_params='labeled')
        #     id_clf.precompute(idx_cand, idx_eval,
        #                       fit_params='all', pred_params='labeled')
        #     if self.candidate_to_labeled:
        #         # idx_train ('labeled'), idx_cand ('all') exists above
        #         # TODO: consider only equal instances would be sufficient
        #         id_clf.precompute(idx_cand, idx_cand,
        #                           fit_params='all', pred_params='all')
        # if self.consider_unlabeled:
        #     id_clf.precompute(idx_train, idx_eval,
        #                       fit_params='labeled', pred_params='unlabeled')
        #     id_clf.precompute(idx_cand, idx_eval,
        #                       fit_params='all', pred_params='unlabeled')

        id_clf = super()._precompute_and_fit_clf(
            id_clf,
            X_full,
            y_full,
            idx_train,
            idx_cand,
            idx_eval,
            fit_clf=fit_clf,
        )

        return id_clf
