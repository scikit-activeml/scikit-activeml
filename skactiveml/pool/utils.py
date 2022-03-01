from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, check_array, \
    check_consistent_length

from ..base import SkactivemlClassifier
from ..classifier import PWC
from ..utils import MISSING_LABEL, is_labeled, is_unlabeled, \
    check_missing_label, check_equal_missing_label, check_type, check_indices

# '__all__' is necessary to create the sphinx docs.
__all__ = ['IndexClassifierWrapper']


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
        ignore_partial_fit=False,
        use_speed_up=False,
        missing_label=MISSING_LABEL,
    ):
        self.clf = clf
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.ignore_partial_fit = ignore_partial_fit
        self.use_speed_up = use_speed_up
        self.missing_label = missing_label

        # Validate classifier type.
        check_type(self.clf, 'clf', SkactivemlClassifier)

        # Check X, y, sample_weight: will be done by base clf
        check_consistent_length(self.X, self.y)

        if sample_weight is not None:
            check_consistent_length(self.X, self.sample_weight)

        # deep copy classifier as it might be fitted already
        check_type(set_base_clf, 'set_base_clf', bool)
        if hasattr(self.clf, "classes_"):
            self.clf_ = deepcopy(self.clf)

            if set_base_clf:
                self.base_clf_ = deepcopy(self.clf_)
        else:
            if set_base_clf:
                raise NotFittedError(
                    'Classifier is not yet fitted but `set_base_clf=True` '
                    'in `__init__` is set to True.'
                )

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

            self.clf_ = deepcopy(self.clf)
            self.clf_.metric = 'precomputed'
            self.clf_.metric_dict = {}


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
        idx_fit = check_array(idx_fit, ensure_2d=False, dtype=int)
        idx_fit = check_indices(idx_fit, self.X, dim=0)

        idx_pred = check_array(idx_pred, ensure_2d=False, dtype=int)
        idx_pred = check_indices(idx_pred, self.X, dim=0)

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
        sample_weight: array-like of shape (n_sub_samples), optional
            (default=None)
            Weights of training samples in `X[idx]`.
        set_base_clf : bool, default=False
            If True, the base classifier will be set to the newly fitted
            classifier

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted IndexClassifierWrapper.

        """
        idx = check_array(idx, ensure_2d=False, dtype=int)
        idx = check_indices(idx, self.X, dim=0)

        check_type(set_base_clf, 'set_base_clf', bool)

        self.idx_ = idx
        if y is None:
            self.y_ = self.y[idx]

            # TODO if self.y_ has no labels, warning
        else:
            y = check_array(y, ensure_2d=False, force_all_finite='allow-nan')
            check_consistent_length(idx, y)
            self.y_ = y

        if sample_weight is None:
            self.sample_weight_ = None if \
                self.sample_weight is None else self.sample_weight[idx]
        else:
            check_consistent_length(sample_weight, y)
            self.sample_weight_ = sample_weight

        if not hasattr(self, 'clf_'):
            self.clf_ = clone(self.clf)

        self.clf_.fit(
            self.get_X(idx), self.y_, self.sample_weight_
        )

        if set_base_clf:
            self.base_clf_ = deepcopy(self.clf_)
            if not self.use_partial_fit:
                self.base_y_ = self.y_.copy()
                self.base_idx_ = self.idx_.copy()

        return self

    # TODO: add replace=False
    def partial_fit(self, idx, y=None, sample_weight=None,
                    use_base_clf=False, set_base_clf=False):
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
            classifier

        Returns
        -------
        self: IndexClassifierWrapper,
            The fitted IndexClassifierWrapper.

        """
        idx = check_array(idx, ensure_2d=False, dtype=int)
        idx = check_indices(idx, self.X, dim=0)

        check_type(use_base_clf, 'use_base_clf', bool)
        check_type(set_base_clf, 'set_base_clf', bool)

        if use_base_clf:
            if not hasattr(self, 'base_clf_'):
                raise NotFittedError(
                    'Base classifier is not set. Please use '
                    '`set_base_clf=True` in `__init__`, `fit`, or '
                    '`partial_fit`.')
        else:
            if not hasattr(self, 'idx_'): # TODO evtl clf_
                raise ValueError('Classifier is not fitted. Please `fit` '
                                 'before using `partial_fit`.')

        if use_base_clf:
            self.y_ = self.base_y_.copy()
        if y is not None:
            y = check_array(y, ensure_2d=False, force_all_finite='allow-nan')
            check_consistent_length(idx, y)
            # TODO what happens if labeled instance gets other label
            self.y_[idx] = y
        else:
            if is_unlabeled(self.y_[idx], self.missing_label_).all():
                raise Warning('New data does not contain labels. Did you '
                              'forget to pass `y` to `partial_fit`?')

        if use_base_clf:
            self.sample_weight_ = self.base_sample_weight_.copy()
        if sample_weight is not None:
            check_consistent_length(sample_weight, y)

            if self.sample_weight_ is None:
                raise ValueError('`sample_weight` cannot be passed if '
                                 '`sample_weight` has not been given in '
                                 '`init` or `fit`.')

            # TODO just overwrite sample_weights if existing?
            self.sample_weight_[idx] = sample_weight

        # TODO: training instances/labels should only appear once?
        #  should be handeled as sets? YES
        # TODO: handle partial_fit as y

        if self.use_partial_fit:
            if use_base_clf:
                self.clf_ = deepcopy(self.base_clf_)
            self.clf_.partial_fit(
                self.X[idx], self.y_[idx], self.sample_weight_[idx]
            )
            # TODO: evtl different behavior cmp to no partial fit

            if set_base_clf:
                self.base_clf_ = deepcopy(self.clf_)
        else:
            if use_base_clf:
                if not hasattr(self, 'base_idx_'):
                    raise ValueError(
                        'Base classifier has been initialized in init without '
                        'passing the indices on which it has been fitted. '
                        'Please fit the base classifier using `fit` and set '
                        '`use_base_clf=True`.'
                    )
                ref_idx = self.base_idx_
                ref_y = self.base_y_
                ref_sw = self.base_sample_weight_
            else:
                ref_idx = self.idx_
                ref_y = self.y_
                ref_sw = self.sample_weight_

            idx_ = np.concatenate([ref_idx, idx], axis=0)
            y_ = np.concatenate([ref_y, y_add], axis=0)
            if ref_sw is not None and sw_add is not None:
                sw_ = np.concatenate([ref_sw, sw_add], axis=0)
            else:
                sw_ = None
                # TODO Warning if different

            self.fit(idx_, y=y_, sample_weight=sw_, set_base_clf=set_base_clf)

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
    #
    # # TODO should these be containted?
    # def get_X(self, idx):
    #     return self.X[idx]
    #
    # def get_y(self, idx):
    #     return self.y[idx]
    #
    # def get_sample_weight(self, idx):
    #     if self.sample_weight is None:
    #         return None
    #     else:
    #         return self.sample_weight[idx]

    def __getattr__(self, item):
        if 'clf_' in self.__dict__:
            return getattr(self.clf_, item)
        else:
            return getattr(self.clf, item)

    #def get_classes(self):
    #    return self.clf_.classes_

    #def get_label_encoder(self):
    #    return self.clf_._le
