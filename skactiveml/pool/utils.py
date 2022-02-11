from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, check_array

from ..base import SkactivemlClassifier
from ..classifier import PWC
from ..utils import MISSING_LABEL, is_labeled, is_unlabeled, \
    check_missing_label, check_equal_missing_label, check_type


class IndexClassifierWrapper:
    """
    Classifier to simplify retraining classifiers in an active learning
    scenario. The idea is to pass all instances at once and use their indices
    to access them. Thereby, optimization is possible e.g. by pre-computing
    kernel-matrices. Moreover, this wrapper implements partial fit for all
    classifiers and includes a base classifier that can be used to simulate adding
    different instance-label pairs to the same classifier.

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
    sample_weight: array-like of shape (n_samples), optional (default=None)
        Weights of training samples in `X`.
    set_base_clf : bool, default=False
        If True, the base classifier will be set to the newly fitted
        classifier
    ignore_partial_fit : bool, optional (default: True)
        Specifies if the `partial_fit` function of `self.clf` should be used
        (if implemented).
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
        ignore_partial_fit=True,
        use_speed_up=True,
        missing_label=MISSING_LABEL,
    ):
        self.clf = clf
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.set_base_clf = set_base_clf
        self.ignore_partial_fit = ignore_partial_fit
        self.use_speed_up = use_speed_up
        self.missing_label = missing_label

        # Validate classifier type.
        check_type(self.clf, 'clf', SkactivemlClassifier)

        # Check X, y, sample_weight: will be done by base clf

        # deep copy classifier as it might be fitted already
        self.clf_ = deepcopy(self.clf)

        # Check set_base_clf
        check_type(self.set_base_clf, 'set_base_clf', bool)

        # Check and use partial fit if applicable
        check_type(self.ignore_partial_fit, 'ignore_partial_fit', bool)
        self.use_partial_fit = (
            hasattr(self.clf, 'partial_fit') and not self.ignore_partial_fit
        )

        # Check use_speed_up
        check_type(self.use_speed_up, 'use_speed_up', bool)

        # Check missing label
        check_missing_label(self.missing_label)
        self.missing_label_ = self.missing_label
        check_equal_missing_label(self.clf.missing_label, self.missing_label_)

        # prepare PWC
        if isinstance(self.clf, PWC) and self.use_speed_up:
            self.pwc_metric_ = self.clf.metric
            self.pwc_metric_dict_ = (
                {} if self.clf.metric_dict is None else self.clf.metric_dict
            )
            self.pwc_K_ = np.full([len(self.X), len(self.X)], np.nan)

            self.clf_.metric = 'precomputed'
            self.clf_.metric_dict = {}

        # initialize base classifier if necessary
        if self.set_base_clf:
            self.base_clf_ = deepcopy(self.clf_)

    def precompute(
        self, idx_fit, idx_pred, fit_params='all', pred_params='all'
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
        # precompute PWC
        if isinstance(self.clf, PWC) and self.use_speed_up:
            if fit_params == 'all':
                idx_fit_ = idx_fit
            elif fit_params == 'labeled':
                idx_fit_ = idx_fit[
                    is_labeled(
                        self.y[idx_fit], missing_label=self.missing_label_
                    )
                ]
            elif fit_params == 'unlabeled':
                idx_fit_ = idx_fit[
                    is_unlabeled(
                        self.y[idx_fit], missing_label=self.missing_label_
                    )
                ]
            else:
                raise ValueError(f'`fit_params`== {fit_params} not defined')

            if pred_params == 'all':
                idx_pred_ = idx_pred
            elif pred_params == 'labeled':
                idx_pred_ = idx_pred[
                    is_labeled(
                        self.y[idx_pred], missing_label=self.missing_label_
                    )
                ]
            elif pred_params == 'unlabeled':
                idx_pred_ = idx_pred[
                    is_unlabeled(
                        self.y[idx_pred], missing_label=self.missing_label_
                    )
                ]
            else:
                raise ValueError(f'`pred_params`== {pred_params} not defined')

            if len(idx_fit_) > 0 and len(idx_pred_) > 0:
                self.pwc_K_[np.ix_(idx_fit_, idx_pred_)] = pairwise_kernels(
                    self.X[idx_fit_],
                    self.X[idx_pred_],
                    self.pwc_metric_,
                    **self.pwc_metric_dict_,
                )

    def fit(self, idx, set_base_clf=False):
        """Fit the model using `self.X[idx]` as training data and `self.y[idx]`
        as class labels.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that will be used to fit the classifier.
        set_base_clf : bool, default=False
            If True, the base classifier will be set to the newly fitted
            classifier

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted IndexClassifierWrapper.

        """
        idx = check_array(idx, ensure_2d=False, dtype=int)
        self.idx_ = idx

        self.clf_.fit(
            self.get_X(idx), self.get_y(idx), self.get_sample_weight(idx)
        )

        if set_base_clf:
            self.base_idx_ = self.idx_.copy()
            self.base_clf_ = deepcopy(self.clf_)

        return self

    def partial_fit(self, idx, y=None,
                    use_base_clf=False, set_base_clf=False):
        """Update the fitted model using additional samples in `self.X[idx]`
        and y as class labels.

        Parameters
        ----------
        idx : array-like of shape (n_sub_samples)
            Indices of samples in `X` that will be used to fit the classifier.
        y : array-like of shape (n_sub_samples)
            Class labels of the training samples corresponding to `X[idx]`.
            Missing labels are represented the attribute 'missing_label'.
        use_base_clf : bool, default=False
            If True, the base classifier will be used to update the fit instead
            of the current classifier. Here, it is necessary that the base
            classifier has been set once.
        set_base_clf : bool, default=False
            If True, the base classifier will be set to the newly fitted
            classifier

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted IndexClassifierWrapper.

        """
        if use_base_clf:
            if not hasattr(self, self.base_clf_):
                raise ValueError('Base classifier has not been initialized. '
                                 'Please set `set_base_clf=True` in `__init__`,'
                                 '`fit`, or `partial_fit`.')
            ref_idx = np.concatenate([self.base_fit_idx, idx], axis=0)
        else:
            ref_idx = np.concatenate([self.idx_, idx], axis=0)
        self.idx_ = np.concatenate([ref_idx, idx], axis=0)

        if y is None:
            y = self.get_y(idx)

        if self.use_partial_fit:
            if use_base_clf:
                self.clf_ = deepcopy(self.base_clf_).partial_fit(
                    self.get_X(idx), y, self.get_sample_weight(idx)
                )
            else:
                self.clf_ = self.clf_.partial_fit(
                    self.get_X(idx), y, self.get_sample_weight(idx)
                )
        else:
            if use_base_clf:
                self.clf_ = clone(self.base_clf_)

            y_new = np.concatenate([self.get_y(ref_idx), y], axis=0)
            self.clf_ = self.clf_.fit(
                self.get_X(self.idx_), y_new, self.get_sample_weight(self.idx_)
            )

        if set_base_clf:
            self.base_idx_ = self.idx_.copy()
            self.base_clf_ = self.clf_

        return self

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
        if isinstance(self.clf, PWC) and self.use_speed_up:
            P = self.clf_.predict_freq(self.pwc_K_[self.idx_, :][:, idx].T)

            # check if results contain NAN
            if np.isnan(P).any():
                raise ValueError(
                    'Error in defining what should be '
                    'pre-computed in PWC. Not all necessary '
                    'information is available which results in '
                    'NaNs in `predict_proba`.'
                )
            return P
        else:
            return self.clf_.predict_freq(self.X[idx])

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
        if isinstance(self.clf, PWC) and self.use_speed_up:
            P = self.clf_.predict_proba(self.pwc_K_[self.idx_, :][:, idx].T)

            # check if results contain NAN
            if np.isnan(P).any():
                raise ValueError(
                    'Error in defining what should be '
                    'pre-computed in PWC. Not all necessary '
                    'information is available which results in '
                    'NaNs in `predict_proba`.'
                )
            return P
        else:
            return self.clf_.predict_proba(self.X[idx])

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
        if isinstance(self.clf, PWC) and self.use_speed_up:
            P = self.clf_.predict(self.pwc_K_[self.idx_, :][:, idx].T)

            # check if results contain NAN
            if np.isnan(P).any():
                raise ValueError(
                    'Error in defining what should be '
                    'pre-computed in PWC. Not all necessary '
                    'information is available which results in '
                    'NaNs in `predict_proba`.'
                )
            return P
        else:
            return self.clf_.predict(self.X[idx])

    def get_X(self, idx):
        return self.X[idx]

    def get_y(self, idx):
        return self.y[idx]

    def get_sample_weight(self, idx):
        if self.sample_weight is None:
            return None
        else:
            return self.sample_weight[idx]

    def __getattr__(self, item):
        return getattr(self.clf_, item)

    #def get_classes(self):
    #    return self.clf_.classes_

    #def get_label_encoder(self):
    #    return self.clf_._le
