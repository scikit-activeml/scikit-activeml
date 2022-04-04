import warnings
from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_array, check_consistent_length

from ..base import SkactivemlClassifier
from ..classifier import ParzenWindowClassifier
from ..utils import (
    MISSING_LABEL,
    is_labeled,
    is_unlabeled,
    check_missing_label,
    check_equal_missing_label,
    check_type,
    check_indices,
)

__all__ = ["IndexClassifierWrapper"]


class IndexClassifierWrapper:
    """
    Classifier to simplify retraining classifiers in an active learning
    scenario. The idea is to pass all instances at once and use their indices
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
    y : array-like of shape (n_samples)
        Labels of the training data set (possibly including unlabeled ones
        indicated by self.missing_label.
    sample_weight : array-like of shape (n_samples), optional (default=None)
        Weights of training samples in `X`.
    set_base_clf : bool, default=False
        If True, the base classifier will be set to the newly fitted
        classifier
    ignore_partial_fit : bool, optional (default: True)
        Specifies if the `partial_fit` function of `self.clf` should be used
        (if implemented).
    enforce_unique_samples : bool, optional (default: False)
        If True, `partial_fit` will not simply append additional samples but
        replace the current labels by the new one. If False, instances might
        appear multiple times if their indices are repeated.
    use_speed_up : bool, optional (default: True)
        Specifies if potentially available speed ups should be used. Currently
        implemented for Parzen Window Classifier.
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
                hasattr(self.clf,
                        "partial_fit") and not self.ignore_partial_fit
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
        idx_fit : array-like of shape (n_fit_samples)
            Indices of samples in `X` that will be used to fit the classifier.
        idx_pred : array-like of shape (n_predict_samples)
            Indices of samples in `X` that the classifier will predict for.
        fit_params : string, optional (default='all')
            Parameter to specify if only a subset of the `idx_fit` indices
            will be used later. Can be of value 'all', 'labeled', or
            'unlabeled'.
        pred_params : string, optional (default='all')
            Parameter to specify if only a subset of the `idx_predict` indices
            will be used later. Can be of value 'all', 'labeled', or
            'unlabeled'.
        """
        idx_fit = check_array(idx_fit, ensure_2d=False, dtype=int)
        idx_fit = check_indices(idx_fit, self.X, dim=0)

        idx_pred = check_array(idx_pred, ensure_2d=False, dtype=int)
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
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that will be used to fit the classifier.
        y : array-like of shape (n_sub_samples), optional (default=None)
            Class labels of the training samples corresponding to `X[idx]`.
            Missing labels are represented the attribute 'missing_label'.
            If None, labels passed in the `init` will be used.
        sample_weight: array-like of shape (n_sub_samples), optional
            (default=None)
            Weights of training samples in `X[idx]`.
            If None, weights passed in the `init` will be used.
        set_base_clf : bool, default=False
            If True, the base classifier will be set to the newly fitted
            classifier

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted IndexClassifierWrapper.

        """
        # check idx
        idx = check_array(idx, ensure_2d=False, dtype=int)
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
            y = check_array(y, ensure_2d=False, force_all_finite="allow-nan")
            check_consistent_length(idx, y)

        # check sample_weight
        if sample_weight is None:
            sample_weight = self._copy_sw(
                self._get_sw(self.sample_weight, idx=idx)
            )
            # TODO deepcopy
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
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
        and y as class labels.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that will be used to fit the classifier.
        y : array-like of shape (n_sub_samples), optional (default=None)
            Class labels of the training samples corresponding to `X[idx]`.
            Missing labels are represented the attribute 'missing_label'.
        sample_weight: array-like of shape (n_sub_samples), optional
            (default=None)
            Weights of training samples in `X[idx]`.
        use_base_clf : bool, default=False
            If True, the base classifier will be used to update the fit instead
            of the current classifier. Here, it is necessary that the base
            classifier has been set once.
        set_base_clf : bool, default=False
            If True, the base classifier will be set to the newly fitted
            classifier.

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted IndexClassifierWrapper.

        """

        # check idx
        add_idx = check_array(idx, ensure_2d=False, dtype=int)
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
                y, ensure_2d=False, force_all_finite="allow-nan"
            )
            check_consistent_length(add_idx, add_y)

        # check sample_weight
        if sample_weight is None:
            add_sample_weight = self._copy_sw(
                self._get_sw(self.sample_weight, idx=add_idx)
            )
        else:
            add_sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(add_idx, add_sample_weight)

        # handle case when partial fit of clf is used
        if self.use_partial_fit:
            if use_base_clf:
                self.clf_ = deepcopy(self.base_clf_)

            # partial fit clf
            self.clf_.partial_fit(self.X[add_idx], add_y, add_sample_weight)

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
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that are to be predicted.

        Returns
        -------
        y : array-like, shape (n_sub_samples)
            Predicted class labels of the input samples.
        """
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            P = self.pwc_K_[self.idx_, :][:, idx].T

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
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that are to be predicted.

        Returns
        -------
        P : array-like, shape (n_sub_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        if isinstance(self.clf, ParzenWindowClassifier) and self.use_speed_up:
            P = self.pwc_K_[self.idx_, :][:, idx].T

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
            P = self.pwc_K_[self.idx_, :][:, idx].T

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
            If True, the result will describe if the base classifier is
            fitted.

        Returns
        -------
        is_fitted : boolean
            Boolean describing if the classifier is fitted.
        """
        clf = "base_clf_" if base_clf else "clf_"
        if clf in self.__dict__:
            return hasattr(getattr(self, clf), "classes_")
        else:
            return False

    def __getattr__(self, item):
        if "clf_" in self.__dict__:
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
