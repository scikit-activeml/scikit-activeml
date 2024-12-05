"""
Wrapper for scikit-learn classifiers to deal with missing labels and labels
from multiple annotators.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>
import warnings
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from sklearn.base import MetaEstimatorMixin, is_classifier
from sklearn.utils import check_consistent_length
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    has_fit_parameter,
)
from skorch import NeuralNet
from torch import nn

from ..base import SkactivemlClassifier
from ..utils import (
    rand_argmin,
    MISSING_LABEL,
    is_labeled,
    check_random_state,
    check_equal_missing_label,
    check_classifier_params,
    check_type,
    check_scalar,
    match_signature,
)


class SklearnClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """SklearnClassifier

    Implementation of a wrapper class for scikit-learn classifiers such that
    missing labels can be handled. Therefore, samples with missing labels are
    filtered.

    Parameters
    ----------
    estimator : sklearn.base.ClassifierMixin with predict_proba method
        scikit-learn classifier that is able to deal with missing labels.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : array-like of shape (n_classes,)
        Holds the label for each class after fitting.
    cost_matrix_ : array-like of shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    estimator_ : sklearn.base.ClassifierMixin with predict_proba method
        The scikit-learn classifier after calling the fit method.
    """

    def __init__(
        self,
        estimator,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.estimator = estimator

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            Missing labels are represented the attribute 'missing_label'.
            In case of multiple labels per sample (i.e., n_outputs > 1), the
            samples are duplicated.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: SklearnClassifier,
            The SklearnClassifier is fitted on the training data.
        """
        return self._fit(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as class
        labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            Missing labels are represented the attribute 'missing_label'.
            In case of multiple labels per sample (i.e., n_outputs > 1), the
            samples are duplicated.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self : SklearnClassifier,
            The SklearnClassifier is fitted on the training data.
        """
        return self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the 'predict' method of the
            'estimator'.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self)
        X = check_array(X, **self.check_X_dict_)
        self._check_n_features(X, reset=False)
        if self.is_fitted_:
            if self.cost_matrix is None:
                y_pred = self.estimator_.predict(X, **predict_kwargs)
            else:
                P = self.predict_proba(X)
                costs = np.dot(P, self.cost_matrix_)
                y_pred = rand_argmin(
                    costs, random_state=self.random_state_, axis=1
                )
        else:
            p = self.predict_proba([X[0]])[0]
            y_pred = self.random_state_.choice(
                np.arange(len(self.classes_)), len(X), replace=True, p=p
            )
            y_pred = self._le.inverse_transform(y_pred)
        y_pred = y_pred.astype(self.classes_.dtype)
        return y_pred

    @match_signature("estimator", "predict_proba")
    def predict_proba(self, X, **predict_proba_kwargs):
        """Return probability estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the 'predict_proba' method of the
            'estimator'.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        check_is_fitted(self)
        X = check_array(X, **self.check_X_dict_)
        self._check_n_features(X, reset=False)
        if self.is_fitted_:
            P = self.estimator_.predict_proba(X, **predict_proba_kwargs)
            # map the predicted classes to self.classes
            if P.shape[1] != len(self.classes_):
                P_ext = np.zeros((len(X), len(self.classes_)))
                est_classes = self.estimator_.classes_
                indices_est = np.where(np.isin(est_classes, self.classes_))[0]
                class_indices = np.searchsorted(
                    self.classes_, est_classes[indices_est]
                )
                P_ext[:, class_indices] = 1 if len(class_indices) == 1 else P
                P = P_ext
            if not np.any(np.isnan(P)):
                return P

        warnings.warn(
            f"Since the 'base_estimator' could not be fitted when"
            f" calling the `fit` method, the class label "
            f"distribution`_label_counts={self._label_counts}` is used to "
            f"make the predictions."
        )
        if sum(self._label_counts) == 0:
            return np.ones([len(X), len(self.classes_)]) / len(self.classes_)
        else:
            return np.tile(
                self._label_counts / np.sum(self._label_counts), [len(X), 1]
            )

    def _fit(self, fit_function, X, y, sample_weight=None, **fit_kwargs):
        # Check input parameters.
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

        # Check whether estimator is a valid classifier.
        if not is_classifier(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "classifier.".format(self.estimator)
            )

        # Check whether estimator can deal with cost matrix.
        if self.cost_matrix is not None and not hasattr(
            self.estimator, "predict_proba"
        ):
            raise ValueError(
                "'cost_matrix' can be only set, if 'estimator'"
                "implements 'predict_proba'."
            )
        if fit_function == "fit" or not hasattr(self, "n_features_in_"):
            self._check_n_features(X, reset=True)
        elif fit_function == "partial_fit":
            self._check_n_features(X, reset=False)
        if hasattr(self, "estimator_"):
            if fit_function != "partial_fit":
                self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = deepcopy(self.estimator)
        # count labels per class
        is_lbld = is_labeled(y, missing_label=-1)
        self._label_counts = [
            np.sum(y[is_lbld] == c) for c in range(len(self._le.classes_))
        ]
        try:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)
            y_lbld_inv = self._le.inverse_transform(y_lbld)
            if np.sum(is_lbld) == 0:
                raise ValueError("There is no labeled data.")
            elif (
                not has_fit_parameter(self.estimator, "sample_weight")
                or sample_weight is None
            ):
                if fit_function == "partial_fit":
                    fit_kwargs["classes"] = self.classes_
                    self.estimator_.partial_fit(
                        X=X_lbld, y=y_lbld_inv, **fit_kwargs
                    )
                elif fit_function == "fit":
                    self.estimator_.fit(X=X_lbld, y=y_lbld_inv, **fit_kwargs)
            else:
                if fit_function == "partial_fit":
                    fit_kwargs["classes"] = self.classes_
                    fit_kwargs["sample_weight"] = sample_weight[is_lbld]
                    self.estimator_.partial_fit(
                        X=X_lbld,
                        y=y_lbld_inv,
                        **fit_kwargs,
                    )
                elif fit_function == "fit":
                    fit_kwargs["sample_weight"] = sample_weight[is_lbld]
                    self.estimator_.fit(
                        X=X_lbld,
                        y=y_lbld_inv,
                        **fit_kwargs,
                    )
            self.is_fitted_ = True
        except Exception as e:
            self.is_fitted_ = False
            warnings.warn(
                "The 'base_estimator' could not be fitted because of"
                " '{}'. Therefore, the class labels of the samples "
                "are counted and will be used to make predictions. "
                "The class label distribution is `_label_counts={}`.".format(
                    e, self._label_counts
                )
            )
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "is_fitted_")

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SlidingWindowClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """SlidingWindowClassifier

    Implementation of a wrapper class for SkactivemlClassifier such that the
    number of training samples can be limited to the latest `window_size`
    samples. Furthermore, saves X, y and sample_weight, enabling the use of a
    partial fit for any classifier.

    Parameters
    ----------
    estimator : sklearn.base.SkactivemlClassifier
        The classifier to be wrapped. If this classifier already implements a
        `partial_fit`, this method will be overwritten by this wrapper using
        the sliding window approach.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    window_size: int, default=None,
        Value to represent the estimator sliding window size for X, y and
        sample weight. If 'None' the window is unrestricted in its size.
    only_labeled: bool, default=False
        If True, unlabeled samples are discarded.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self,
        estimator,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        window_size=None,
        only_labeled=False,
        random_state=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.estimator = estimator
        self.only_labeled = only_labeled
        self.window_size = window_size

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            Missing labels are represented the attribute 'missing_label'.
            In case of multiple labels per sample (i.e., n_outputs > 1), the
            samples are duplicated.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: SlidingWindowClassifier,
            The SlidingWindowClassifier is fitted on the training data.
        """
        # Check whether estimator is a valid classifier.
        if not isinstance(self.estimator, SkactivemlClassifier):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier"
                "classifier.".format(self.estimator)
            )
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

        self._add_samples("fit", X, y, sample_weight)
        X_train = np.array(self.X_train_)
        y_train = np.array(self.y_train_)
        sample_weight_train = None
        if self.sample_weight_train_ is not None:
            sample_weight_train = np.array(
                self.sample_weight_train_, dtype=float
            )
        return self._fit(
            X=X_train,
            y=y_train,
            sample_weight=sample_weight_train,
            **fit_kwargs,
        )

    @match_signature("estimator", "fit")
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as class
        labels. If 'base_estimator' has no partial_fit function use fit with
        the sliding window for X, y and sample_weight.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            Missing labels are represented the attribute 'missing_label'.
            In case of multiple labels per sample (i.e., n_outputs > 1), the
            samples are duplicated.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self : SlidingWindowClassifier,
            The SlidingWindowClassifier is fitted on the training data.
        """
        # Check whether estimator is a valid classifier.
        if not isinstance(self.estimator, SkactivemlClassifier):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier.".format(self.estimator)
            )
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

        self._add_samples("partial_fit", X, y, sample_weight)
        X_train = np.array(self.X_train_)
        y_train = np.array(self.y_train_)
        sample_weight_train = None
        if self.sample_weight_train_ is not None:
            sample_weight_train = np.array(
                self.sample_weight_train_, dtype=float
            )
        return self._fit(
            X=X_train,
            y=y_train,
            sample_weight=sample_weight_train,
            **fit_kwargs,
        )

    def _add_samples(self, fit_func, X, y, sample_weight=None):
        if not hasattr(self, "X_train_"):
            self.X_train_ = deque(maxlen=self.window_size)
        if not hasattr(self, "y_train_"):
            self.y_train_ = deque(maxlen=self.window_size)
        if not hasattr(self, "sample_weight_train_"):
            self.sample_weight_train_ = deque(maxlen=self.window_size)
        if self.only_labeled:
            is_lbld = is_labeled(y, self.missing_label)
            X = X[is_lbld]
            y = y[is_lbld]
            if sample_weight is not None:
                sample_weight = sample_weight[is_lbld]
            else:
                sample_weight = None
        # reset the window if fit is called otherwise extend the window with
        # the given data
        if fit_func == "fit":
            self.X_train_ = deque(maxlen=self.window_size)
            self.y_train_ = deque(maxlen=self.window_size)
            self.sample_weight_train_ = deque(maxlen=self.window_size)
        self.X_train_.extend(X)
        self.y_train_.extend(y)
        if sample_weight is not None:
            self.sample_weight_train_.extend(sample_weight)
        else:
            self.sample_weight_train_ = None

    def _fit(self, X, y, sample_weight=None, **fit_kwargs):
        # Check whether estimator can deal with cost matrix.
        if self.cost_matrix is not None and not hasattr(
            self.estimator, "predict_proba"
        ):
            raise ValueError(
                "'cost_matrix' can be only set, if 'estimator'"
                "implements 'predict_proba'."
            )

        self._check_n_features(X, reset=True)

        if hasattr(self, "estimator_"):
            self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = deepcopy(self.estimator)

        if has_fit_parameter(self.estimator, "sample_weight"):
            fit_kwargs["sample_weight"] = sample_weight

        self.estimator_.fit(X=X, y=y, **fit_kwargs)

        return self

    def _validate_data(self, X, y, sample_weight=None, check_X_dict=None):
        # super._validate_data is not called because training with partial fit
        # with only one single available class in y leads to an error if
        # self.classes is not set, even though self.classes has no function in
        # this class.
        if self.window_size is not None:
            check_scalar(
                self.window_size,
                "window_size",
                int,
                min_val=0,
                min_inclusive=False,
            )
        check_type(self.only_labeled, "only_labeled", bool)

        check_y_dict = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "ensure_2d": False,
            "force_all_finite": False,
            "dtype": None,
        }

        # Check input parameters.
        y = check_array(y, **check_y_dict)
        if len(y) == 0:
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f"`y` has the shape {y.shape} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"identical shapes."
                )

        # Check common classifier parameters.
        check_classifier_params(
            self.classes, self.missing_label, self.cost_matrix
        )

        if (
            self.cost_matrix is not None
            and self.estimator.cost_matrix is not None
            and not np.array_equiv(
                self.cost_matrix, self.estimator.cost_matrix
            )
        ):
            raise ValueError(
                "'cost_matrix' and estimator.cost_matrix must be equal. "
                "Got {} is not equal to {}.".format(
                    self.cost_matrix, self.estimator.cost_matrix
                )
            )
        # self.missing_label is not testet completly and
        # needs to be checked for the general test.
        # if general test is removed, remove this check.
        _ = is_labeled(y, missing_label=self.missing_label)

        check_equal_missing_label(
            self.missing_label,
            self.estimator.missing_label,
        )
        # if self.classes=None or self.estimator.classes=None then no checks
        # are done if general test is removed it should be checked again
        if (
            self.classes is not None
            and self.estimator.classes is not None
            and not np.array_equiv(self.classes, self.estimator.classes)
        ):
            raise ValueError(
                "'classes' and estimator.classes must be equal. "
                "Got {} is not equal to {}.".format(
                    self.classes, self.estimator.classes
                )
            )

        # Store and check random state.
        self.random_state_ = check_random_state(self.random_state)

        return X, y, sample_weight

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the 'predict' method of the
            'estimator'.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self)
        X = check_array(X, **self.check_X_dict_)
        self._check_n_features(X, reset=False)
        return self.estimator_.predict(X, **predict_kwargs)

    @match_signature("estimator", "predict_proba")
    def predict_proba(self, X, **predict_proba_kwargs):
        """Return probability estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the 'predict_proba' method of the
            'estimator'.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        check_is_fitted(self)
        X = check_array(X, **self.check_X_dict_)
        self._check_n_features(X, reset=False)
        proba = self.estimator_.predict_proba(X, **predict_proba_kwargs)
        return proba

    @match_signature("estimator", "predict_freq")
    def predict_freq(self, X, **predict_freq_kwargs):
        """Return class frequency estimates for the test samples `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test samples whose class frequencies are to be estimated.

        Returns
        -------
        F: array-like of shape (n_samples, classes)
            The class frequency estimates of the test samples 'X'. Classes are
            ordered according to attribute 'classes_'.
        """
        check_is_fitted(self)
        X = check_array(X, **self.check_X_dict_)
        self._check_n_features(X, reset=False)
        freq = self.estimator_.predict_freq(X, **predict_freq_kwargs)
        return freq

    def __getattr__(self, item):
        if "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")


class SkorchClassifier(NeuralNet, SkactivemlClassifier):
    """SkorchClassifier

    Implement a wrapper class, to make it possible to use `PyTorch` with
    `skactiveml`. This is achieved by providing a wrapper around `PyTorch`
    that has a skactiveml interface and also be able to handle missing labels.
    This wrapper is based on the open-source library `skorch` [1].

    Parameters
    ----------
    module : torch module (class or instance)
      A PyTorch :class:`~torch.nn.Module`. In general, the
      uninstantiated class should be passed, although instantiated
      modules will also work.
    criterion : torch criterion (class), default: nn.NLLoss()
      The uninitialized criterion (loss) used to optimize the
      module.
    *args: arguments
        more possible arguments for initialize your neural network
        see: https://skorch.readthedocs.io/en/stable/net.html
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    **kwargs : keyword arguments
        more possible parameters to customizing your neural network
        see: https://skorch.readthedocs.io/en/stable/net.html

    References
    ----------
    [1] Marian Tietz, Thomas J. Fan, Daniel Nouri, Benjamin Bossan, and
    skorch Developers. skorch: A scikit-learn compatible neural network
    library that wraps PyTorch, July 2017.
    """

    def __init__(
        self,
        module,
        criterion=nn.NLLLoss(),
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        **kwargs,
    ):
        super(SkorchClassifier, self).__init__(
            module,
            criterion,
            **kwargs,
        )

        SkactivemlClassifier.__init__(
            self,
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )

    def fit(self, X, y, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
        y : array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label)
        fit_params : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: SkorchClassifier,
            The SkorchClassifier is fitted on the training data.
        """

        # check input parameters
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            check_X_dict=self.check_X_dict_,
        )

        is_lbld = is_labeled(y, missing_label=self.missing_label)
        if np.sum(is_lbld) == 0:
            raise ValueError("There is no labeled data.")
        else:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)
            return super(SkorchClassifier, self).fit(
                X_lbld, y_lbld, **fit_params
            )

    def predict(self, X):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        return SkactivemlClassifier.predict(self, X)
