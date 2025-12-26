"""
Wrapper for scikit-learn classifiers to deal with missing labels and labels
from multiple annotators.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>
import warnings
from collections import deque
from copy import deepcopy

import numpy as np
from sklearn.base import MetaEstimatorMixin, is_classifier
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    has_fit_parameter,
)

from sklearn.utils import check_consistent_length
from sklearn.exceptions import NotFittedError

from ..base import SkactivemlClassifier
from ..utils import (
    rand_argmin,
    MISSING_LABEL,
    ExtLabelEncoder,
    is_labeled,
    check_random_state,
    check_equal_missing_label,
    check_classifier_params,
    check_type,
    check_scalar,
    match_signature,
    check_n_features,
)

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.base import SkorchMixin
    from skactiveml.utils import make_criterion_tuple_aware

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


class SklearnClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """Sklearn Classifier

    Implementation of a wrapper class for `scikit-learn` classifiers such that
    missing labels can be handled. Therefore, samples with missing labels are
    filtered.

    Parameters
    ----------
    estimator : sklearn.base.ClassifierMixin with predict_proba method
        The `scikit-learn` classifier to be wrapped.
    include_unlabeled_samples : bool, default=False
        - If `False`, only labeled samples are passed to the `fit` method of
          the `estimator`.
        - If `True`, all samples including the unlabeled ones are passed to
          the `fit` method of the `estimator`. Ensure that your `estimator`
          is able to handle unlabeled samples marked by `missing_label`.
          Otherwise, `missing_label` is interpreted as a regular class label.
          Note that semi-supervised classifiers of `sklearn` expect
          `missing_label=-1`.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If `None`, the classes are determined
        during `fit`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None`.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        Holds the label for each class after fitting.
    cost_matrix_ : numpy.ndarray of shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    estimator_ : sklearn.base.ClassifierMixin with predict_proba method
        The scikit-learn classifier after calling the `fit` method.
    """

    def __init__(
        self,
        estimator,
        include_unlabeled_samples=False,
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
        self.include_unlabeled_samples = include_unlabeled_samples

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples. Missing
            labels are represented the attribute `self.missing_label_`. In case
            of multiple labels per sample (i.e., n_outputs > 1), the samples
            are duplicated.
        sample_weight : array-like of shape (n_samples,) or\
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as `y`.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self: SklearnClassifier,
            The `SklearnClassifier` fitted on the training data.
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
        """Partially fitting the model using `X` as training data and `y` as
        class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples. Missing
            labels are represented the attribute `self.missing_label_`. In case
            of multiple labels per sample (i.e., n_outputs > 1), the samples
            are duplicated.
        sample_weight : array-like of shape (n_samples,) or\
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as `y`.
        fit_kwargs : dict-like
            Further parameters as input to the `partial_fit` method of the
            `estimator`.

        Returns
        -------
        self : SklearnClassifier,
            The `SklearnClassifier` is fitted on the training data.
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
        """Return class label predictions for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the `predict` method of the
            `estimator`.

        Returns
        -------
        y_pred :  numpy.ndarray of shape (n_samples,)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
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
        """Return probability estimates for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the `predict_proba` method of the
            `estimator`.

        Returns
        -------
        P : array-like of shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            according to the attribute `self.classes_`.
        """
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
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
            reset=fit_function == "fit" or not hasattr(self, "n_features_in_"),
        )

        # Check whether estimator is a valid classifier.
        if not is_classifier(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "classifier.".format(self.estimator)
            )

        # Check boolean flag.
        check_type(
            self.include_unlabeled_samples,
            "include_unlabeled_samples",
            bool,
        )

        # Check whether estimator can deal with cost matrix.
        if self.cost_matrix is not None and not hasattr(
            self.estimator, "predict_proba"
        ):
            raise ValueError(
                "'cost_matrix' can be only set, if 'estimator'"
                "implements 'predict_proba'."
            )
        if hasattr(self, "estimator_"):
            if fit_function != "partial_fit":
                self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = deepcopy(self.estimator)
        # count labels per class
        if self.include_unlabeled_samples:
            is_included = np.full_like(y, fill_value=True, dtype=bool)
        else:
            is_included = is_labeled(y, missing_label=-1)
        self._label_counts = [
            np.sum(y[is_included] == c) for c in range(len(self._le.classes_))
        ]
        try:
            X_train = X[is_included]
            y_train = y[is_included].astype(np.int64)
            y_train_inv = self._le.inverse_transform(y_train)
            if np.sum(is_included) == 0:
                raise ValueError("There is no labeled data.")
            elif (
                not has_fit_parameter(self.estimator, "sample_weight")
                or sample_weight is None
            ):
                if fit_function == "partial_fit":
                    fit_kwargs["classes"] = self.classes_
                    self.estimator_.partial_fit(
                        X=X_train, y=y_train_inv, **fit_kwargs
                    )
                elif fit_function == "fit":
                    self.estimator_.fit(X=X_train, y=y_train_inv, **fit_kwargs)
            else:
                if fit_function == "partial_fit":
                    fit_kwargs["classes"] = self.classes_
                    fit_kwargs["sample_weight"] = sample_weight[is_included]
                    self.estimator_.partial_fit(
                        X=X_train,
                        y=y_train_inv,
                        **fit_kwargs,
                    )
                elif fit_function == "fit":
                    fit_kwargs["sample_weight"] = sample_weight[is_included]
                    self.estimator_.fit(X=X_train, y=y_train_inv, **fit_kwargs)
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
        if hasattr(self, "is_fitted_"):
            return True

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False

        # set attributes that would be set by the fit function
        self.is_fitted_ = True
        self.estimator_ = deepcopy(self.estimator)
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        return True

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SlidingWindowClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """Sliding Window Classifier

    Implementation of a wrapper class for `SkactivemlClassifier` such that the
    number of training samples can be limited to the latest `window_size`
    samples. Furthermore, saves `X`, `y` and `sample_weight`, enabling the use
    of a `partial_fit` for any classifier.

    Parameters
    ----------
    estimator : sklearn.base.SkactivemlClassifier
        The classifier to be wrapped. If this classifier already implements a
        `partial_fit`, this method will be overwritten by this wrapper using
        the sliding window approach.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If `None`, `classes` are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    window_size : int, default=None,
        Value to represent the estimator sliding window size for X, y and
        sample weight. If `None` the window is unrestricted in its size.
    only_labeled : bool, default=False
        If `True`, unlabeled samples are discarded.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
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
        """Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples. Missing
            labels are represented the attribute `self.missing_label_`. In case
            of multiple labels per sample (i.e., n_outputs > 1), the samples
            are duplicated.
        sample_weight : array-like of shape (n_samples,) or\
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as `y`.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self: SlidingWindowClassifier,
            The `SlidingWindowClassifier` is fitted on the training data.
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
        """Partially fitting the model using `X` as training data and `y` as
        class labels. If `base_estimator` has no `partial_fit` function use
        `fit` with the sliding window for X, y and sample_weight.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples. Missing
            labels are represented the attribute `self.missing_label_`. In case
            of multiple labels per sample (i.e., n_outputs > 1), the samples
            are duplicated.
        sample_weight : array-like of shape (n_samples,) or\
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as `y`.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

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
            "ensure_all_finite": False,
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
        """Return class label predictions for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the `predict` method of the
            `estimator`.

        Returns
        -------
        y_pred : numpy.ndarray shape (n_samples,)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self)
        return self.estimator_.predict(X, **predict_kwargs)

    @match_signature("estimator", "predict_proba")
    def predict_proba(self, X, **predict_proba_kwargs):
        """Return probability estimates for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the `predict_proba` method of the
            `estimator`.

        Returns
        -------
        P : numpy.ndarray shape (n_samples, classes)
            The class probabilities of the input samples `X`. Classes are
            ordered according to the attribute `self.classes_`.
        """
        check_is_fitted(self)
        proba = self.estimator_.predict_proba(X, **predict_proba_kwargs)
        return proba

    @match_signature("estimator", "predict_freq")
    def predict_freq(self, X, **predict_freq_kwargs):
        """Return class frequency estimates for the test samples `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Test samples whose class frequencies are to be estimated.

        Returns
        -------
        F : numpy.ndarray of shape (n_samples, classes)
            The class frequency estimates of the test samples `X`. Classes are
            ordered according to the attribute `self.classes_`.
        """
        check_is_fitted(self)
        freq = self.estimator_.predict_freq(X, **predict_freq_kwargs)
        return freq

    def __getattr__(self, item):
        if "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")


if successful_skorch_torch_import:

    class SkorchClassifier(SkactivemlClassifier, SkorchMixin):
        """SkorchClassifier

        Implement a classification wrapper class to make it possible to use
        `torch` with `skactiveml`. This is achieved by providing a wrapper
        around `torch` that has a `skactiveml` interface and can handle
        missing labels. This wrapper is based on the open-source library
        `skorch` [1]_.

        Notes
        -----
        Adjust your `criterion` and `module.forward` outputs consistently.
        See the documentation of the parameters `forward_outputs` and
        `criterion_output_keys` for further details.

        Parameters
        ----------
        module : torch.nn.Module.__class__ or torch.nn.Module
            A PyTorch `torch.nn.Module`. In general, the uninstantiated class
            should be passed, although instantiated modules will also work.
        criterion : torch.nn.Module or torch.nn.Module.__class__, \
                default=torch.nn.CrossEntropyLoss
            The loss (criterion) used to optimize the module.

            - If a class (subclass of `torch.nn.Module`) is passed
              (e.g. `torch.nn.CrossEntropyLoss`), it is instantiated
              internally.
            - If an instance is passed (e.g. `torch.nn.CrossEntropyLoss()`),
              that instance (or a wrapped copy of it) is used.

            By default, `torch.nn.CrossEntropyLoss` is used as criterion.
        forward_outputs : dict[str, tuple[int, Callable | None]] or None,\
                default=None
            Dictionary that describes how to get and post-process the outputs
            of `module.forward` for prediction. This parameter replaces the
            functionality of `predict_nonlinearity` in a `skorch.net.NeuralNet`
            (see documentation of `neural_net_param_dict`).

            Given `raw_outputs = module.forward(x)`, each entry
            `name -> (idx, transform)` in `forward_outputs` is interpreted as:

            - `idx` : int
              Index into `raw_outputs` (0-based).
            - `transform` : callable or `None`
              If not `None`, it is applied to the selected raw tensor
              `raw_outputs[idx]`. Otherwise, the raw tensor is used.

            This allows multiple named outputs to reference the same raw tensor
            with different transforms, for example::

                forward_outputs = {
                    "proba":  (0, torch.nn.Softmax(dim=-1)),  # probabilities
                    "logits": (0, None),                      # raw scores
                    "emb":    (1, None),                      # embeddings
                }

            The first entry in `forward_outputs` defines the primary
            scores used for prediction:

            - In `predict_proba`, the transformed first output is
              interpreted as class probabilities `P`.
            - In `predict`, the class probabilities `P` returned by
              `predict_proba` are used to infer class label predictions.

            If `forward_outputs` is `None`, a sensible default is chosen
            for common single-output classifiers based on the `criterion`:

            - If `criterion` is `torch.nn.CrossEntropyLoss`, it is
              assumed that `module.forward` returns logits and the
              effective mapping is::

                  {"proba": (0, torch.nn.Softmax(dim=-1))}

            - If `criterion` is `torch.nn.NLLLoss`, it is assumed that
              `module.forward` returns log-probabilities and the effective
              mapping is::

                  {"proba": (0, torch.exp)}

            - For all other criteria, a single-output module is assumed to
              already produce values in probability space, and the effective
              mapping is::

                  {"proba": (0, None)}

        criterion_output_keys : str or sequence of str or None, default=None
            Name or names of the forward outputs that are passed to the
            loss / criterion during training. Use this when
            `module.forward` returns multiple outputs
            (e.g. `(logits, embeddings, ...)`), but the criterion expects
            a single tensor input or a specific tuple of inputs.

            The names must refer to keys of the effective `forward_outputs`
            mapping. If `criterion_output_keys` is not `None` and
            `forward_outputs` is `None`, a `ValueError` is raised
            because the names cannot be resolved.

            - If a `str`, the corresponding named output of
              `module.forward` (i.e., the raw tensor selected via its
              index in `forward_outputs` before applying the transform)
              is passed to the criterion (e.g. `"logits"` to use only the
              class scores).
            - If a sequence of `str`, the selected named outputs are passed to
              the criterion in that order. Each raw forward output index may
              appear at most once: using multiple names that resolve to the
              same underlying index (e.g. `"proba"` and `"logits"` both
              pointing to index 0) is not allowed and results in a
              `ValueError`.
            - If `None`, the first output defined by the effective
              `forward_outputs` mapping is used as criterion input.

            To pass all distinct forward outputs to the criterion in the
            same order as `forward_outputs`, choose one representative name
            per raw output index and set, for example::

                # assuming that each key refers to a different raw index
                criterion_output_keys = tuple(forward_outputs.keys())

            If `forward_outputs` contains multiple names that refer to the
            same raw output index (aliases such as `"proba"` and `"logits"`
            both mapping to index 0), you must select at most one name per
            raw index in `criterion_output_keys`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added. `module`, `criterion`, and `predict_nonlinearity` are not
            allowed in this dictionary.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved. The encoded label data
            type is always  `np.int64`.
        include_unlabeled_samples : bool, default=False
            - If `False`, only labeled samples are passed to the `fit` method
              of the estimator.
            - If `True`, all samples including the unlabeled ones are passed to
              the `fit` method of the estimator. Ensure that the `criterion`
              is able to handle unlabeled samples marked by `missing_label`.
              Otherwise, `missing_label` is interpreted as a regular class
              label.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during the fit.
        missing_label : scalar or str or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes)
            Cost matrix with `cost_matrix[i, j]` indicating the cost of
            predicting class `classes[j]` for a sample of class
            `classes[i]`. Can only be set if `classes` is not `None`.
        random_state : int or RandomState instance or None, default=None
            Determines random number generation for methods that rely on
            randomness (e.g. `predict` for stochastic models). Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Marian Tietz, Thomas J. Fan, Daniel Nouri, Benjamin Bossan, and
           skorch Developers. skorch: A scikit-learn compatible neural network
           library that wraps PyTorch, July 2017.
        """

        def __init__(
            self,
            module,
            criterion=nn.CrossEntropyLoss,
            forward_outputs=None,
            criterion_output_keys=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            include_unlabeled_samples=False,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(SkorchClassifier, self).__init__(
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
            )
            self.module = module
            self.criterion = criterion
            self.forward_outputs = forward_outputs
            self.criterion_output_keys = criterion_output_keys
            self.neural_net_param_dict = neural_net_param_dict
            self.sample_dtype = sample_dtype
            self.include_unlabeled_samples = include_unlabeled_samples

        def fit(self, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless `warm_start` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)
            fit_params : dict-like
                Further parameters as input to the 'fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier,
                `SkorchClassifier` object fitted on the training data.
            """
            return self._fit("fit", X, y, **fit_params)

        def partial_fit(self, X, y, **fit_params):
            """Fit the module without re-initialization.

            If the module was already initialized, by calling `partial_fit`,
            the module will not be re-initialized again.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by `self.missing_label`)
            fit_params : dict-like
                Further parameters as input to the 'partial_fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier
                `SkorchClassifier` object fitted on the training data.
            """
            return self._fit("partial_fit", X, y, **fit_params)

        def predict(self, X, extra_outputs=None):
            """Return class predictions for the test samples `X`.

            By default, this method returns only the predicted classes
            `y_pred`. The predictions are obtained via the class probabilities
            `P` outputted by `predict_proba`. If `extra_outputs` is provided,
            a tuple is returned whose first element is `y_pred` and whose
            remaining elements are the requested additional forward outputs,
            in the order specified by `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or or sequence of str, default=None
                Names of additional outputs to return next to `y_pred`. The
                names must be a subset of the keys of the effective
                `forward_outputs` mapping.

                For example, if::

                    self.forward_outputs = {
                        "proba":  (0, torch.nn.Softmax(dim=-1)),
                        "logits": (0, None),
                        "emb":    (1, None),
                    }

                then valid values for `extra_outputs` include `"emb"` or
                `["emb", "logits"]`.

                - If `extra_outputs is None`, only `y_pred` is returned.
                - If `extra_outputs` is a string, e.g. `"emb"`, the
                  return value is `(y_pred, emb)`.
                - If `extra_outputs` is a sequence of strings, the return
                  value is `(y_pred, out_1, out_2, ...)`, where `out_i`
                  corresponds to the i-th name in `extra_outputs`.

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,)
                Predicted class labels of the test samples.
            *extras : numpy.ndarray, optional
                Additional outputs. Only present if `extra_outputs` is not
                `None`. In that case, the method returns a single tuple whose
                first element is `y_pred` and whose remaining elements
                (`extras`) correspond to the requested forward outputs in the
                order given by `extra_outputs`.
            """
            return super().predict(
                X=X,
                extra_outputs=extra_outputs,
            )

        def predict_proba(self, X, extra_outputs=None):
            """Return class probability estimates for the test samples `X`.

            By default, this method returns only the predicted class
            probabilities `P`. If `extra_outputs` is provided, a tuple is
            returned whose first element is `y_pred` and whose remaining
            elements are the requested additional forward outputs, in the
            order specified by `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to `P`. The names
                must be a subset of the keys of the effective `forward_outputs`
                mapping.

                For example, if::

                    self.forward_outputs = {
                        "proba":  (0, torch.nn.Softmax(dim=-1)),
                        "logits": (0, None),
                        "emb":    (1, None),
                    }

                then valid values for `extra_outputs` include `"emb"` or
                `["emb", "logits"]`.

                - If `extra_outputs is None`, only `P` is returned.
                - If `extra_outputs` is a string, e.g. `"logits"`, the
                  return value is `(P, logits)`.
                - If `extra_outputs` is a sequence of strings, the return
                  value is `(P, out_1, out_2, ...)`, where `out_i`
                  corresponds to the i-th name in `extra_outputs`.

            Returns
            -------
            P : numpy.ndarray of shape (n_samples, n_classes)
                Class probabilities of the test samples. Classes are ordered
                according to `self.classes_`.
            *extras : numpy.ndarray, optional
                Additional outputs. Only present if `extra_outputs` is not
                `None`. In that case, the method returns a single tuple whose
                first element is `P` and whose remaining elements
                (`extras`) correspond to the requested forward outputs in the
                order given by `extra_outputs`.
            """
            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Check input parameters.
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )

            # Resolve effective forward_outputs (either user-provided or
            # defaulted based on the criterion).
            forward_outputs = self._effective_forward_outputs()

            # Forward propagation whose return values depends on the request
            # ones.
            fw_out = self._forward_with_named_outputs(
                X, forward_outputs=forward_outputs, extra_outputs=extra_outputs
            )

            # First element is expected to be the class probabilities.
            P = fw_out[0] if isinstance(fw_out, tuple) else fw_out
            self._initialize_fallbacks(P)
            return fw_out

        def _effective_forward_outputs(self):
            """Return the effective `forward_outputs` mapping.

            If the user did not specify `forward_outputs`, choose a reasonable
            default for common criteria (e.g., `nn.CrossEntropyLoss`) and a
            simple single-output module.

            The returned mapping has the form::

                {name: (idx, transform)}

            where `idx` is the index into the tuple returned by
            `module.forward` (0-based) and `transform` is a callable or
            `None`. For the defaults below, a single-output module is assumed,
            i.e., `idx == 0`.
            """
            # User explicitly provided a mapping: trust it.
            if self.forward_outputs is not None:
                return self.forward_outputs

            # No explicit mapping: handle common single-output cases.
            crit_cls = (
                self.criterion
                if isinstance(self.criterion, type)
                else self.criterion.__class__
            )

            if crit_cls is nn.CrossEntropyLoss:
                # Single-output network returning logits.
                return {"proba": (0, nn.Softmax(dim=-1))}

            if crit_cls is nn.NLLLoss:
                # Module returns log-probabilities.
                return {"proba": (0, torch.exp)}

            # Fallback: treat the single forward output as already in
            # probability space. Caller is responsible for making this true.
            return {"proba": (0, None)}

        def _net_parts(self, X=None, y=None):
            """Assemble and validate network components.

            Implementations should perform any optional checks or normalization
            of constructor/init parameters (e.g., shape consistency, dtype
            checks, wrapping criteria), then return the ready-to-use pieces for
            `skorch.NeuralNet`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples for optional validation.
            y : array-like of shape (n_samples, ...), default=None
                Target values for optional validation.

            Returns
            -------
            module : torch.nn.Module.__class__ or torch.nn.Module
                A PyTorch `torch.nn.Module`. In general, the uninstantiated
                class should be passed, although instantiated modules will also
                work.
            criterion : torch.nn.Module.__class__
                The uninitialized criterion (loss) used to optimize the module.
            params : dict
                Keyword arguments (excluding `predict_non_linearity`) for
                `skorch.NeuralNet` construction. Must be a mapping and may be
                empty.
            """
            criterion = self.criterion
            criterion = make_criterion_tuple_aware(
                criterion=criterion,
                criterion_output_keys=self.criterion_output_keys,
                forward_outputs=self._effective_forward_outputs(),
            )
            return (
                self.module,
                criterion,
                self.neural_net_param_dict or {},
            )

        def _validate_data_kwargs(self):
            """Return kwargs forwarded to `_validate_data`.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by `_validate_data`.
            """
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": self.sample_dtype,
            }
            check_type(
                self.include_unlabeled_samples,
                "include_unlabeled_samples",
                bool,
            )
            return {"check_X_dict": self.check_X_dict_}

        def _return_training_data(self, X, y):
            """Return only samples and labels required for training.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Targets with unlabeled entries following the subclass'
                convention.

            Returns
            -------
            X_train : ndarray or None
                Training samples or `None` if none exist.
            y_train : ndarray or None
                Training labels or `None` if none exist.
            """
            X_train, y_train = None, None
            if self.include_unlabeled_samples:
                is_included = np.full_like(y, fill_value=True, dtype=bool)
            else:
                is_included = is_labeled(y, missing_label=-1)
            if np.sum(is_included) > 0:
                X_train = X[is_included]
                y_train = y[is_included].astype(np.int64)
            return X_train, y_train

        def _initialize_fallbacks(self, P):
            """Initialize label/cost fallbacks if the classifier was not fitted
            before.

            Parameters
            ----------
            P : array-like of shape (n_samples, n_classes)
                Class-probability array used only to infer `n_classes` when
                `self.classes` is `None`.
            """
            self.random_state_ = check_random_state(self.random_state)
            if not hasattr(self, "_le"):
                self._le = ExtLabelEncoder(
                    classes=self.classes, missing_label=self.missing_label
                )
                if self.classes is not None:
                    y_dummy = self.classes
                else:
                    y_dummy = np.arange(P.shape[-1], dtype=int)
                self._le.fit(y_dummy)
                self.classes_ = self._le.classes_
            if not hasattr(self, "cost_matrix_"):
                self.cost_matrix_ = (
                    1 - np.eye(len(self.classes_))
                    if self.cost_matrix is None
                    else self.cost_matrix
                )
