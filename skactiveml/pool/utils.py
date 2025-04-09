import warnings
from copy import deepcopy

import numpy as np
import scipy
from scipy import integrate
from scipy.special import roots_hermitenorm
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_array, check_consistent_length

from ..base import (
    SkactivemlClassifier,
    ProbabilisticRegressor,
)
from ..classifier import ParzenWindowClassifier
from ..utils import (
    MISSING_LABEL,
    is_labeled,
    is_unlabeled,
    check_missing_label,
    check_equal_missing_label,
    check_type,
    check_indices,
    check_random_state,
    check_scalar,
)

__all__ = ["IndexClassifierWrapper"]

from ..utils._validation import _check_callable


class IndexClassifierWrapper:
    """
    Classifier to simplify retraining classifiers in an active learning
    scenario. The idea is to pass all samples at once and use their indices
    to access them. Thereby, optimization is possible e.g. by pre-computing
    kernel-matrices. Moreover, this wrapper implements partial fit for all
    classifiers and includes a base classifier that can be used to simulate
    adding different instance-label pairs to the same classifier.

    Parameters
    ----------
    clf : skactiveml.base.SkactivemlClassifier
        The base classifier implementing the methods `fit` and `predict_proba`.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples,)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `self.missing_label`).
    sample_weight : array-like of shape (n_samples,), default=None
        Weights of training samples in `X`.
    set_base_clf : bool, default=False
        If `True`, the base classifier will be set to the newly fitted
        classifier.
    ignore_partial_fit : bool, default=True
        Specifies if the `partial_fit` function of `self.clf` should be used
        (if implemented).
    enforce_unique_samples : bool, default=False
        If `True`, `partial_fit` will not simply append additional samples but
        replace the current labels by the new one. If `False`, samples might
        appear multiple times if their indices are repeated.
    use_speed_up : bool, default=True
        Specifies if potentially available speed ups should be used. Currently
        implemented for `skactiveml.classifier.ParzenWindowClassifier`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    """

    def __init__(
        self,
        clf,
        X,
        y,
        sample_weight=None,
        set_base_clf=False,
        ignore_partial_fit=False,
        enforce_unique_samples=False,
        use_speed_up=False,
        missing_label=MISSING_LABEL,
    ):
        self.clf = clf
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.ignore_partial_fit = ignore_partial_fit
        self.enforce_unique_samples = enforce_unique_samples
        self.use_speed_up = use_speed_up
        self.missing_label = missing_label

        # Validate classifier type.
        check_type(self.clf, "clf", SkactivemlClassifier)

        # Check X, y, sample_weight: will be done by base clf
        self.X = check_array(self.X, allow_nd="True")
        self.y = check_array(
            self.y,
            ensure_2d=False,
            ensure_all_finite=False,
            dtype=None,
        )
        check_consistent_length(self.X, self.y)

        if self.sample_weight is not None:
            check_consistent_length(self.X, self.sample_weight)

        check_type(set_base_clf, "set_base_clf", bool)

        # deep copy classifier as it might be fitted already
        if hasattr(self.clf, "classes_"):
            self.clf_ = deepcopy(self.clf)

            if set_base_clf:
                self.base_clf_ = deepcopy(self.clf_)
        else:
            if set_base_clf:
                raise NotFittedError(
                    "Classifier is not yet fitted but `set_base_clf=True` "
                    "in `__init__` is set to True."
                )

        # Check and use partial fit if applicable
        check_type(self.ignore_partial_fit, "ignore_partial_fit", bool)
        self.use_partial_fit = (
            hasattr(self.clf, "partial_fit") and not self.ignore_partial_fit
        )

        check_type(self.enforce_unique_samples, "enforce_unique_samples", bool)
        self.enforce_unique_samples = (
            "check_unique" if enforce_unique_samples else False
        )
        # TODO better change check_indices function

        if self.use_partial_fit and self.enforce_unique_samples:
            warnings.warn(
                "The `partial_fit` function by sklearn might not "
                "ensure that every sample is used only once in the "
                "fitting process."
            )

        # Check use_speed_up
        check_type(self.use_speed_up, "use_speed_up", bool)

        # Check missing label
        check_missing_label(self.missing_label)
        self.missing_label_ = self.missing_label
        if not np.issubdtype(type(self.missing_label), self.y.dtype):
            raise TypeError(
                f"`missing_label` has type {type(missing_label)}, "
                f"which is not compatible with {self.y.dtype} as the "
                f"type of `y`."
            )
        check_equal_missing_label(self.clf.missing_label, self.missing_label_)

        # prepare ParzenWindowClassifier
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            self.pwc_metric_ = self.clf.metric
            self.pwc_metric_dict_ = (
                {} if self.clf.metric_dict is None else self.clf.metric_dict
            )
            self.pwc_K_ = np.full([len(self.X), len(self.X)], np.nan)

            self.clf_ = clone(self.clf)
            self.clf_.metric = "precomputed"
            self.clf_.metric_dict = {}

    def precompute(
        self, idx_fit, idx_pred, fit_params="all", pred_params="all"
    ):
        """
        Function to describe for which samples we should precompute something.
        Will be internally handled differently for different classifiers. The
        function consists of pairs of `idx_fit` and `idx_predict` to describe
        which sequences of fitting and predicting are to be expected.

        Parameters
        ----------
        idx_fit : array-like of shape (n_fit_samples,)
            Indices of samples in `X` that will be used to fit the classifier.
        idx_pred : array-like of shape (n_predict_samples,)
            Indices of samples in `X` that the classifier will predict for.
        fit_params : string, default='all'
            Parameter to specify if only a subset of the `idx_fit` indices
            will be used later. Can be of value 'all', 'labeled', or
            'unlabeled'.
        pred_params : string, default='all'
            Parameter to specify if only a subset of the `idx_predict` indices
            will be used later. Can be of value 'all', 'labeled', or
            'unlabeled'.
        """
        idx_fit = check_array(
            idx_fit, ensure_2d=False, dtype=int, input_name="`idx_fit`"
        )
        idx_fit = check_indices(idx_fit, self.X, dim=0)

        idx_pred = check_array(
            idx_pred, ensure_2d=False, dtype=int, input_name="`idx_pred`"
        )
        idx_pred = check_indices(idx_pred, self.X, dim=0)

        # precompute ParzenWindowClassifier
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            if fit_params == "all":
                idx_fit_ = idx_fit
            elif fit_params == "labeled":
                idx_fit_ = idx_fit[
                    is_labeled(
                        self.y[idx_fit], missing_label=self.missing_label_
                    )
                ]
            elif fit_params == "unlabeled":
                idx_fit_ = idx_fit[
                    is_unlabeled(
                        self.y[idx_fit], missing_label=self.missing_label_
                    )
                ]
            else:
                raise ValueError(f"`fit_params`== {fit_params} not defined")

            if pred_params == "all":
                idx_pred_ = idx_pred
            elif pred_params == "labeled":
                idx_pred_ = idx_pred[
                    is_labeled(
                        self.y[idx_pred], missing_label=self.missing_label_
                    )
                ]
            elif pred_params == "unlabeled":
                idx_pred_ = idx_pred[
                    is_unlabeled(
                        self.y[idx_pred], missing_label=self.missing_label_
                    )
                ]
            else:
                raise ValueError(f"`pred_params`== {pred_params} not defined")

            if len(idx_fit_) > 0 and len(idx_pred_) > 0:
                self.pwc_K_[np.ix_(idx_fit_, idx_pred_)] = pairwise_kernels(
                    self.X[idx_fit_],
                    self.X[idx_pred_],
                    self.pwc_metric_,
                    **self.pwc_metric_dict_,
                )

    def fit(self, idx, y=None, sample_weight=None, set_base_clf=False):
        """Fit the model using `self.X[idx]` as training data and `self.y[idx]`
        as class labels.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples,)
            Indices of samples in `X` that will be used to fit the classifier.
        y : array-like of shape (n_sub_samples,), default=None
            Class labels of the training samples corresponding to `X[idx]`.
            Missing labels are represented the attribute 'missing_label'.
            If `None`, labels passed in the `init` will be used.
        sample_weight : array-like of shape (n_sub_samples,), default=None
            Weights of training samples in `X[idx]`.
            If `None`, weights passed in the `init` will be used.
        set_base_clf : bool, default=False
            If `True`, the base classifier will be set to the newly fitted
            classifier.

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted `IndexClassifierWrapper`.
        """
        # check idx
        idx = check_array(idx, ensure_2d=False, dtype=int, input_name="`idx`")
        idx = check_indices(
            idx, self.X, dim=0, unique=self.enforce_unique_samples
        )

        # check set_base_clf
        check_type(set_base_clf, "set_base_clf", bool)

        # check y
        if y is None:
            y = self.y[idx]
            if is_unlabeled(y, missing_label=self.missing_label_).all():
                warnings.warn("All labels are of `missing_label` in `fit`.")
        else:
            y = check_array(
                y,
                ensure_2d=False,
                ensure_all_finite=False,
                dtype=self.y.dtype,
                input_name="`y`",
            )
            check_consistent_length(idx, y)

        # check sample_weight
        if sample_weight is None:
            sample_weight = self._copy_sw(
                self._get_sw(self.sample_weight, idx=idx)
            )
            # TODO deepcopy
        else:
            sample_weight = check_array(
                sample_weight, ensure_2d=False, input_name="`sample_weight`"
            )
            check_consistent_length(sample_weight, y)

        # check if a clf_ exists
        if "clf_" not in self.__dict__:
            self.clf_ = clone(self.clf)

        # fit classifier
        self.clf_.fit(self.X[idx], y, sample_weight)

        # store data for further processing
        if not self.use_partial_fit:
            self.idx_ = idx
            self.y_ = y
            self.sample_weight_ = sample_weight

        # set base clf if necessary
        if set_base_clf:
            self.base_clf_ = deepcopy(self.clf_)
            if not self.use_partial_fit:
                self.base_idx_ = self.idx_.copy()
                self.base_y_ = self.y_.copy()
                self.base_sample_weight_ = self._copy_sw(self.sample_weight_)

        return self

    def partial_fit(
        self,
        idx,
        y=None,
        sample_weight=None,
        use_base_clf=False,
        set_base_clf=False,
    ):
        """Update the fitted model using additional samples in `self.X[idx]`
        and `y` as class labels.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples,)
            Indices of samples in `X` that will be used to fit the classifier.
        y : array-like of shape (n_sub_samples,), default=None
            Class labels of the training samples corresponding to `X[idx]`.
            Missing labels are represented by the attribute `missing_label`.
        sample_weight : array-like of shape (n_sub_samples,), default=None
            Weights of training samples in `X[idx]`.
        use_base_clf : bool, default=False
            If `True`, the base classifier will be used to update the fit
            instead of the current classifier. Here, it is necessary that the
            base classifier has been set once.
        set_base_clf : bool, default=False
            If `True`, the base classifier will be set to the newly fitted
            classifier.

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted `IndexClassifierWrapper`.
        """

        # check idx
        add_idx = check_array(
            idx, ensure_2d=False, dtype=int, input_name="`add_idx`"
        )
        add_idx = check_indices(
            add_idx, self.X, dim=0, unique=self.enforce_unique_samples
        )

        # check use_base_clf
        check_type(use_base_clf, "use_base_clf", bool)

        if use_base_clf:
            if not self.is_fitted(base_clf=True):
                raise NotFittedError(
                    "Base classifier is not set. Please use "
                    "`set_base_clf=True` in `__init__`, `fit`, or "
                    "`partial_fit`."
                )
        else:
            if not self.is_fitted(base_clf=False):
                raise NotFittedError(
                    "Classifier is not fitted. Please `fit` before using "
                    "`partial_fit`."
                )

        # check set_base_clf
        check_type(set_base_clf, "set_base_clf", bool)

        # check y
        if y is None:
            add_y = self.y[add_idx]
            if is_unlabeled(add_y, missing_label=self.missing_label_).all():
                warnings.warn(
                    "All labels are of `missing_label` in " "`partial_fit`."
                )
        else:
            add_y = check_array(
                y,
                ensure_2d=False,
                ensure_all_finite=False,
                dtype=self.y.dtype,
                input_name="`y`",
            )
            check_consistent_length(add_idx, add_y)

        # check sample_weight
        if sample_weight is None:
            add_sample_weight = self._copy_sw(
                self._get_sw(self.sample_weight, idx=add_idx)
            )
        else:
            add_sample_weight = check_array(
                sample_weight, ensure_2d=False, input_name="`sample_weight`"
            )
            check_consistent_length(add_idx, add_sample_weight)

        # handle case when partial fit of clf is used
        if self.use_partial_fit:
            if use_base_clf:
                self.clf_ = deepcopy(self.base_clf_)

            # partial fit clf
            if add_sample_weight is None:
                self.clf_.partial_fit(self.X[add_idx], add_y)
            else:
                self.clf_.partial_fit(
                    self.X[add_idx], add_y, sample_weight=add_sample_weight
                )

            if set_base_clf:
                self.base_clf_ = deepcopy(self.clf_)

        # handle case using regular fit from clf
        else:
            if not hasattr(self, "idx_"):
                raise NotFittedError(
                    "Fitted classifier from `init` cannot be "
                    "used for `partial_fit` as it is unknown "
                    "where it has been fitted on."
                )
            if use_base_clf:
                self.clf_ = clone(self.base_clf_)
                self.idx_ = self.base_idx_.copy()
                self.y_ = self.base_y_.copy()
                self.sample_weight_ = self._copy_sw(self.base_sample_weight_)

            if self.enforce_unique_samples:
                cur_idx = np.array([i not in add_idx for i in self.idx_])
            else:
                cur_idx = np.arange(len(self.idx_))
            self.idx_ = np.concatenate([self.idx_[cur_idx], add_idx], axis=0)
            self.y_ = np.concatenate([self.y_[cur_idx], add_y], axis=0)
            self.sample_weight_ = self._concat_sw(
                self._get_sw(self.sample_weight_, cur_idx), add_sample_weight
            )

            self.fit(
                self.idx_,
                y=self.y_,
                sample_weight=self.sample_weight_,
                set_base_clf=set_base_clf,
            )

        return self

    def predict(self, idx):
        """Return class label predictions for the input data `X[idx]`.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples,)
            Indices of samples in `X` that are to be predicted.

        Returns
        -------
        y : array-like of shape (n_sub_samples,)
            Predicted class labels of the input samples.
        """
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            if hasattr(self, "idx_"):
                P = self.pwc_K_[self.idx_, :][:, idx].T
            else:
                warnings.warn("Speed-up not possible when prefitted")
                return self.clf.predict_proba(self.X[idx])

            # check if results contain NAN
            if np.isnan(P).any():
                raise ValueError(
                    "Error in defining what should be "
                    "pre-computed in ParzenWindowClassifier. "
                    "Not all necessary "
                    "information is available which results in "
                    "NaNs in `predict_proba`."
                )
            return self.clf_.predict(P)
        else:
            return self.clf_.predict(self.X[idx])

    def predict_proba(self, idx):
        """Return probability estimates for the input data `X[idx]`.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples,)
            Indices of samples in `X` that are to be predicted.

        Returns
        -------
        P : array-like of shape (n_sub_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            if hasattr(self, "idx_"):
                P = self.pwc_K_[self.idx_, :][:, idx].T
            else:
                warnings.warn("Speed-up not possible when prefitted")
                return self.clf.predict_proba(self.X[idx])

            # check if results contain NAN
            if np.isnan(P).any():
                raise ValueError(
                    "Error in defining what should be "
                    "pre-computed in ParzenWindowClassifier. "
                    "Not all necessary "
                    "information is available which results in "
                    "NaNs in `predict_proba`."
                )
            return self.clf_.predict_proba(P)
        else:
            return self.clf_.predict_proba(self.X[idx])

    def predict_freq(self, idx):
        """Return class frequency estimates for the input samples 'X[idx]'.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that are to be predicted.

        Returns
        -------
        F: array-like of shape (n_sub_samples, classes)
            The class frequency estimates of the input samples. Classes are
            ordered according to `classes_`.
        """
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            if hasattr(self, "idx_"):
                P = self.pwc_K_[self.idx_, :][:, idx].T
            else:
                warnings.warn("Speed-up not possible when prefitted")
                return self.clf.predict_proba(self.X[idx])

            # check if results contain NAN
            if np.isnan(P).any():
                raise ValueError(
                    "Error in defining what should be "
                    "pre-computed in ParzenWindowClassifier. "
                    "Not all necessary "
                    "information is available which results in "
                    "NaNs in `predict_proba`."
                )
            return self.clf_.predict_freq(P)
        else:
            return self.clf_.predict_freq(self.X[idx])

    def is_fitted(self, base_clf=False):
        """Returns if the classifier (resp. the base classifier) is fitted.

        Parameters
        ----------
        base_clf : bool, default=False
            If `True`, the result will describe if the base classifier is
            fitted.

        Returns
        -------
        is_fitted : bool
            Flag describing whether the classifier is fitted.
        """
        clf = "base_clf_" if base_clf else "clf_"
        if clf in self.__dict__:
            return hasattr(getattr(self, clf), "classes_")
        else:
            return False

    def __getattr__(self, item):
        if "clf_" in self.__dict__ and hasattr(self.clf_, item):
            return getattr(self.clf_, item)
        else:
            return getattr(self.clf, item)

    def _get_sw(self, sample_weight, idx=None):
        if sample_weight is None:
            return None
        else:
            return sample_weight[idx]

    def _copy_sw(self, sample_weight):
        if sample_weight is None:
            return None
        else:
            return sample_weight.copy()

    def _concat_sw(self, sample_weight, sample_weight_add):
        if sample_weight is None and sample_weight_add is None:
            return None
        if sample_weight is not None and sample_weight_add is not None:
            return np.concatenate([sample_weight, sample_weight_add], axis=0)
        else:
            raise ValueError(
                "All `sample_weight` must be either None or " "given."
            )


def _cross_entropy(
    X_eval, true_reg, other_reg, integration_dict=None, random_state=None
):
    """Calculates the cross entropy.

    Parameters
    ----------
    X_eval : array-like of shape (n_samples, n_features)
        The samples where the cross entropy should be evaluated.
    true_reg: ProbabilisticRegressor
        True distribution of the cross entropy.
    other_reg: ProbabilisticRegressor
        Evaluated distribution of the cross entropy.
    integration_dict: dict, optional default = None
        Dictionary for integration arguments, i.e. `integration method` etc.
        For details see method `conditional_expect`.
    random_state : int | np.random.RandomState, optional
        Random state for cross entropy calculation.

    Returns
    -------
    cross_ent : numpy.ndarray of shape (n_samples)
        The cross entropy.
    """

    if integration_dict is None:
        integration_dict = {}

    check_type(integration_dict, "integration_dict", dict)
    check_type(true_reg, "true_reg", ProbabilisticRegressor)
    check_type(other_reg, "other_reg", ProbabilisticRegressor)
    random_state = check_random_state(random_state)

    dist = _reshape_scipy_dist(
        other_reg.predict_target_distribution(X_eval), shape=(len(X_eval), 1)
    )

    cross_ent = -expected_target_val(
        X_eval,
        dist.logpdf,
        reg=true_reg,
        random_state=random_state,
        **integration_dict,
        vector_func="both",
    )

    return cross_ent


def _update_reg(
    reg,
    X,
    y,
    y_update,
    sample_weight=None,
    idx_update=None,
    X_update=None,
    mapping=None,
):
    """Update the regressor by the updating samples, depending on
    the mapping. Chooses `X_update` if `mapping is None` and updates
    `X[mapping[idx_update]]` otherwise.

    Parameters
    ----------
    reg : SkactivemlRegressor
        The regressor to be updated.
    X : array-like of shape (n_samples, n_features)
        Training data set.
    y : array-like of shape (n_samples)
        Labels of the training data set.
    y_update : array-like of shape (n_updates) or numeric
        Updating labels or updating label.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weight of the training data set. If
    idx_update : array-like of shape (n_updates) or int
        Index of the samples or sample to be updated.
    X_update : array-like of shape (n_updates, n_features) or (n_features,)
        Samples to be updated or sample to be updated.
    mapping : array-like of shape (n_candidates,), default=None
        The deciding mapping.

    Returns
    -------
    reg_new : SkaktivemlRegressor
        The updated regressor.
    """

    if sample_weight is not None and mapping is None:
        raise ValueError(
            "If `sample_weight` is not `None` a mapping "
            "between candidates and the training dataset must "
            "exist."
        )

    if mapping is not None:
        if isinstance(idx_update, (int, np.integer)):
            check_indices([idx_update], A=mapping, unique="check_unique")
        else:
            check_indices(idx_update, A=mapping, unique="check_unique")
        X_new, y_new = _update_X_y(
            X, y, y_update, idx_update=mapping[idx_update]
        )
    else:
        X_new, y_new = _update_X_y(X, y, y_update, X_update=X_update)

    if sample_weight is None:
        reg_new = clone(reg).fit(X_new, y_new)
    else:
        reg_new = clone(reg).fit(X_new, y_new, sample_weight)
    return reg_new


def _update_X_y(X, y, y_update, idx_update=None, X_update=None):
    """Update the training data by the updating samples/labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set.
    y : array-like of shape (n_samples,)
        Labels of the training data set.
    idx_update : array-like of shape (n_updates,) or int
        Index of the samples or sample to be updated.
    X_update : array-like of shape (n_updates, n_features) or (n_features,)
        Samples to be updated or sample to be updated.
    y_update : array-like of shape (n_updates,) or numeric
        Updating label(s).

    Returns
    -------
    X_new : np.ndarray of shape (n_new_samples, n_features)
        The new training data set.
    y_new : np.ndarray of shape (n_new_samples)
        The new labels.
    """

    X = check_array(X, input_name="`X`")
    y = column_or_1d(
        check_array(
            y, ensure_all_finite=False, ensure_2d=False, input_name="`y`"
        )
    )
    check_consistent_length(X, y)

    if isinstance(y_update, (int, float)):
        y_update = np.array([y_update])
    else:
        y_update = check_array(
            y_update,
            ensure_all_finite=False,
            ensure_2d=False,
            ensure_min_samples=0,
            input_name="`y`",
        )
        y_update = column_or_1d(y_update)

    if idx_update is not None:
        if isinstance(idx_update, (int, np.integer)):
            idx_update = np.array([idx_update])
        idx_update = check_indices(idx_update, A=X, unique="check_unique")
        check_consistent_length(y_update, idx_update)
        X_new = X.copy()
        y_new = y.copy()
        y_new[idx_update] = y_update
        return X_new, y_new
    elif X_update is not None:
        X_update = check_array(
            X_update, ensure_2d=False, input_name="`X_update`"
        )
        if X_update.ndim == 1:
            X_update = X_update.reshape(1, -1)
        check_consistent_length(X.T, X_update.T)
        check_consistent_length(y_update, X_update)
        X_new = np.append(X, X_update, axis=0)
        y_new = np.append(y, y_update, axis=0)
        return X_new, y_new
    else:
        raise ValueError("`idx_update` or `X_update` must not be `None`")


def _reshape_scipy_dist(dist, shape):
    """Reshapes the parameters `loc`, `scale`, `df` of a distribution, if they
    exist.

    Parameters
    ----------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The distribution.
    shape : tuple
        The new shape.

    Returns
    -------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The reshaped distribution.
    """
    check_type(dist, "dist", scipy.stats._distn_infrastructure.rv_frozen)
    check_type(shape, "shape", tuple)
    for idx, item in enumerate(shape):
        check_type(item, f"shape[{idx}]", int)

    for argument in ["loc", "scale", "df"]:
        if argument in dist.kwds:
            # check if shapes are compatible
            dist.kwds[argument].shape = shape

    return dist


def expected_target_val(X, target_func, reg, **kwargs):
    """Calculates the conditional expectation of a function depending only on
    the target value for each sample in `X`, i.e.,
    `E[target_func(Y)|X=x]`, where `Y | X=x ~ reg.predict_target_distribution`,
    for `x` in `X`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples where the expectation should be evaluated.
    target_func : callable
        The function that transforms the random variable.
    reg : ProbabilisticRegressor
        Predicts the target distribution over which the expectation is
        calculated.

    Other Parameters
    ----------------
    method : string, optional, optional (default='gauss_hermite')
        The method by which the expectation is computed.

        - 'assume_linear' assumes E[func(Y)|X=x_eval] ~= func(E[Y|X=x_eval])
          and thereby only takes the function value at the expected y value.
        - 'monte_carlo' Basic monte carlo integration. Taking the average
          of randomly drawn samples. `n_integration_samples` specifies the
          number of monte carlo samples.
        - 'quantile' Uses the quantile function to transform the integration
          space into the interval from 0 to 1 and than uses the method from
          'quantile_method' to calculate the integral. The number of
          integration points is specified by `n_integration_samples`.
        - 'gauss_hermite' Uses Gauss-Hermite quadrature. This assumes Y | X
          to be gaussian distributed. The number of evaluation  points is given
          by `n_integration_samples`.
        - 'dynamic_quad' uses `scipy's` function `expect` on the
          `rv_continuous` random variable of `reg`, which in turn uses a
          dynamic gaussian quadrature routine for calculating the integral.
          Performance is worse using a vector function.
    quantile_method : string, default='quadrature'
        Specifies the integration methods used after the quantile
        transformation.

        - 'trapezoid' Trapezoidal method for integration using evenly spaced
          samples.
        - 'simpson' Simpson method for integration using evenly spaced samples.
        - 'average' Taking the average value for integration using evenly
          spaced samples.
        - 'romberg' Romberg method for integration. If `n_integration_samples`
          is not equal to `2**k + 1` for a natural number k, the number of
          samples used for integration is put to the smallest such number
          greater than `n_integration_samples`.
        - 'quadrature' Gaussian quadrature method for integration.
    n_integration_samples : int, default=10
        The number of integration samples used in 'quantile', 'monte_carlo' and
        'gauss-hermite'.
    quad_dict : dict, default=None
        Further arguments for using `scipy's` `expect`
    random_state : int or np.random.RandomState or None, default=None
        Random state for fixing the number generation.
    target_func : bool
        If `True` only the target values will be passed to `func`.
    vector_func : bool or str, default=False
        If `vector_func` is `True`, the integration values are passed as a
        whole to the function `func`. If `vector_func` is 'both', the
        integration values might or might not be passed as a whole. The
        integration values if passed as a whole are of the form (n_samples,
        n_integration), where n_integration denotes the number of integration
        values.

    Returns
    -------
    expectation : numpy.ndarray of shape (n_samples,)
        The conditional expectation for each value applied.
    """

    _check_callable(target_func, "target_func", n_positional_parameters=1)

    def arg_filtered_func(idx_y, x_y, y):
        return target_func(y)

    return _conditional_expect(X, arg_filtered_func, reg, **kwargs)


def _conditional_expect(
    X,
    func,
    reg,
    method=None,
    quantile_method=None,
    n_integration_samples=10,
    quad_dict=None,
    random_state=None,
    vector_func=False,
):
    """Calculates the conditional expectation of a function depending on the
    target value the corresponding feature value and an index for each sample
    in `X`, i.e. E[func(Y, x, idx)|X=x], where
    Y | X=x ~ reg.predict_target_distribution, for x in `X`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples where the expectation should be evaluated.
    func : callable
        The function that transforms the random variable. The signature of the
        function must be of the form `func(y, x, idx)`, where `y` is the target
        value, `x` is the feature value and `idx` is such that `X[idx] = x`.
    reg: ProbabilisticRegressor
        Predicts the target distribution over which the expectation is
        calculated.
    method: string, optional, default='gauss_hermite'
        The method by which the expectation is computed.

        -'assume_linear' assumes E[func(Y)|X=x_eval] ~= func(E[Y|X=x_eval]) and
          thereby only takes the function value at the expected y value.
        -'monte_carlo' Basic monte carlo integration. Taking the average
          of randomly drawn samples. `n_integration_samples` specifies the
          number of monte carlo samples.
        -'quantile' Uses the quantile function to transform the integration
          space into the interval from 0 to 1 and than uses the method from
          'quantile_method' to calculate the integral. The number of
          integration points is specified by `n_integration_samples`.
        -'gauss_hermite' Uses Gauss-Hermite quadrature. This assumes Y | X
          to be gaussian distributed. The number of evaluation  points is given
          by `n_integration_samples`.
        -'dynamic_quad' uses `scipy's` function `expect` on the `rv_continuous`
          random variable of `reg`, which in turn uses a dynamic gaussian
          quadrature routine for calculating the integral. Performance is worse
          using a vector function.
    quantile_method : string, default='quadrature'
        Specifies the integration methods used after the quantile
        transformation.

        - 'trapezoid': Trapezoidal method for integration using evenly spaced
          samples.
        - 'simpson': Simpson method for integration using evenly spaced
          samples.
        - 'average': Taking the average value for integration using evenly
          spaced samples.
        - 'romberg': Romberg method for integration. If `n_integration_samples`
          is not equal to `2**k + 1` for a natural number k, the number of
          samples used for integration is put to the smallest such number
          greater than `n_integration_samples`.
        -'quadrature': Gaussian quadrature method for integration.
    n_integration_samples : int, default=10
        The number of integration samples used in 'quantile', 'monte_carlo' and
        'gauss-hermite'.
    quad_dict : dict, default=None
        Further arguments for using `scipy's` `expect`
    random_state : int or np.random.RandomState or None,default=None
        Random state for fixing the number generation.
    vector_func : bool or str, default=False
        If `vector_func` is `True`, the integration values are passes
        in vectorized form to `func`. If `vector_func` is 'both', the
        integration values might or might not be passed in vectorized form,
        depending what is more efficient. The integration values
        are passed in vectorized form, means that in a call like
        `func(y, x, idx)` `y` is of the form `(n_samples,
        n_integration_samples)`, `x` equals `X` and `idx` is an index map of
        `X`.

    Returns
    -------
    expectation : numpy.ndarray of shape (n_samples)
        The conditional expectation for each value applied.
    """

    X = check_array(X, allow_nd=True, input_name="`X`")

    check_type(reg, "reg", ProbabilisticRegressor)
    check_type(
        method,
        "method",
        target_vals=[
            "monte_carlo",
            "assume_linear",
            "dynamic_quad",
            "gauss_hermite",
            "quantile",
            None,
        ],
    )
    check_type(
        quantile_method,
        "quantile_method",
        target_vals=[
            "trapezoid",
            "simpson",
            "average",
            "romberg",
            "quadrature",
            None,
        ],
    )
    check_scalar(n_integration_samples, "n_monte_carlo", int, min_val=1)
    check_type(quad_dict, "scipy_args", dict, target_vals=[None])
    check_type(vector_func, "vector_func", bool, target_vals=["both"])
    _check_callable(func, "func", n_positional_parameters=3)

    if method is None:
        method = "gauss_hermite"
    if quantile_method is None:
        quantile_method = "quadrature"
    if quad_dict is None:
        quad_dict = {}
    if method == "quantile" and quantile_method == "romberg":
        # n_integration_samples need to be of the form 2**k + 1
        n_integration_samples = (
            2 ** int(np.log2(n_integration_samples) + 1) + 1
        )
    is_optional = vector_func == "both"
    if is_optional:
        vector_func = True

    random_state = check_random_state(random_state)

    def evaluate_func(inner_potential_y):
        if vector_func:
            inner_output = func(np.arange(len(X)), X, inner_potential_y)
        else:
            inner_output = np.zeros_like(inner_potential_y)
            for idx_x, inner_x in enumerate(X):
                for idx_y, y_val in enumerate(inner_potential_y[idx_x]):
                    inner_output[idx_x, idx_y] = func(idx_x, inner_x, y_val)
        return inner_output

    expectation = np.zeros(len(X))

    if method in ["assume_linear", "monte_carlo"]:
        if method == "assume_linear":
            potential_y = reg.predict(X).reshape(-1, 1)
        else:  # method equals "monte_carlo"
            potential_y = reg.sample_y(
                X=X,
                n_samples=n_integration_samples,
                random_state=random_state,
            )
        expectation = np.average(evaluate_func(potential_y), axis=1)
    elif method == "quantile":
        if quantile_method in ["trapezoid", "simpson", "average", "romberg"]:
            eval_points = np.arange(1, n_integration_samples + 1) / (
                n_integration_samples + 1
            )
            cond_dist = _reshape_scipy_dist(
                reg.predict_target_distribution(X), shape=(-1, 1)
            )
            potential_y = cond_dist.ppf(eval_points.reshape(1, -1))
            output = evaluate_func(potential_y)

            if quantile_method == "trapezoid":
                expectation = integrate.trapezoid(
                    output, dx=1 / n_integration_samples, axis=1
                )
            elif quantile_method == "simpson":
                expectation = integrate.simpson(
                    output, dx=1 / n_integration_samples, axis=1
                )
            elif quantile_method == "average":
                expectation = np.average(output, axis=-1)
            else:  # quantile_method equals "romberg"
                expectation = integrate.romb(
                    output, dx=1 / n_integration_samples, axis=1
                )
        else:  # quantile_method equals "quadrature"

            def fixed_quad_function_wrapper(inner_eval_points):
                inner_cond_dist = _reshape_scipy_dist(
                    reg.predict_target_distribution(X), shape=(-1, 1)
                )
                inner_potential_y = inner_cond_dist.ppf(
                    inner_eval_points.reshape(1, -1)
                )

                return evaluate_func(inner_potential_y)

            expectation, _ = integrate.fixed_quad(
                fixed_quad_function_wrapper, 0, 1, n=n_integration_samples
            )
    elif method == "gauss_hermite":
        unscaled_potential_y, weights = roots_hermitenorm(
            n_integration_samples
        )
        cond_mean, cond_std = reg.predict(X, return_std=True)
        potential_y = (
            cond_std[:, np.newaxis] * unscaled_potential_y[np.newaxis, :]
            + cond_mean[:, np.newaxis]
        )
        output = evaluate_func(potential_y)
        expectation = (
            1
            / (2 * np.pi) ** (1 / 2)
            * np.sum(weights[np.newaxis, :] * output, axis=1)
        )
    else:  # method equals "dynamic_quad"
        for idx, x in enumerate(X):
            cond_dist = reg.predict_target_distribution([x])

            def quad_function_wrapper(y):
                if is_optional or not vector_func:
                    return func(idx, x, y)
                else:
                    return func(np.arange(len(X)), X, np.full((len(X), 1), y))[
                        idx
                    ]

            expectation[idx] = cond_dist.expect(
                quad_function_wrapper,
                **quad_dict,
            )

    return expectation
