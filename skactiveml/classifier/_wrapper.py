"""
Wrapper for scikit-learn classifiers to deal with missing labels and labels
from multiple annotators.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>


import warnings
from copy import deepcopy
from collections import deque

import numpy as np
from sklearn.base import MetaEstimatorMixin, is_classifier
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    has_fit_parameter,
)
from sklearn.neighbors import KernelDensity

from ..base import SkactivemlClassifier, ClassFrequencyEstimator
from ..utils import rand_argmin, MISSING_LABEL, check_scalar


class SklearnClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """SklearnClassifier

    Implementation of a wrapper class for scikit-learn classifiers such that
    missing labels can be handled. Therefor, samples with missing labels are
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

    @if_delegate_has_method(delegate="estimator")
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

    @if_delegate_has_method(delegate="estimator")
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
            if P.shape[1] != len(self.classes_):
                P_ext = np.zeros((len(X), len(self.classes_)))
                class_indices = np.asarray(self.estimator_.classes_, dtype=int)
                # Exception for the MLPCLassifier
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
        if (
            not has_fit_parameter(self.estimator, "sample_weight")
            and sample_weight is not None
        ):
            warnings.warn(
                f"{self.estimator} does not support `sample_weight`. "
                f"Therefore, this parameter will be ignored."
            )
        if hasattr(self, "estimator_"):
            if fit_function != "partial_fit":
                self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = deepcopy(self.estimator)
        # Transform in case of 2-dimensional array of class labels.
        if y.ndim == 2:
            X = np.repeat(X, np.size(y, axis=1), axis=0)
            y = y.ravel()
            if sample_weight is not None:
                sample_weight = sample_weight.ravel()
        # count labels per class
        is_lbld = ~np.isnan(y)
        self._label_counts = [
            np.sum(y[is_lbld] == c) for c in range(len(self._le.classes_))
        ]
        try:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)
            if np.sum(is_lbld) == 0:
                raise ValueError("There is no labeled data.")
            elif (
                not has_fit_parameter(self.estimator, "sample_weight")
                or sample_weight is None
            ):
                if fit_function == "partial_fit":
                    classes = self._le.transform(self.classes_)
                    self.estimator_.partial_fit(
                        X=X_lbld, y=y_lbld, classes=classes, **fit_kwargs
                    )
                elif fit_function == "fit":
                    self.estimator_.fit(X=X_lbld, y=y_lbld, **fit_kwargs)
            else:
                if fit_function == "partial_fit":
                    classes = self._le.transform(self.classes_)
                    self.estimator_.partial_fit(
                        X=X_lbld,
                        y=y_lbld,
                        classes=classes,
                        sample_weight=sample_weight[is_lbld],
                        **fit_kwargs,
                    )
                elif fit_function == "fit":
                    self.estimator_.fit(
                        X=X_lbld,
                        y=y_lbld,
                        sample_weight=sample_weight[is_lbld],
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

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class KernelFrequencyClassifier(SklearnClassifier, ClassFrequencyEstimator):
    """KernelFrequencyClassifier

    Implementation of a wrapper class for scikit-learn classifiers such that
    missing labels can be handled. Therefor, samples with missing labels are
    filtered. Furthermore, use a seperate frequency estimator to estimate
    class frequencies, such that the scikit-learn classifier can be used as a
    ClassFrequencyEstimator.

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
    frequency_estimator: sklearn.base.BaseEstimator, default=None
        Model which implements the score_samples method which calculates the
        frequency of observed labels in the neighborhood. if set to None use
        sklear.neighbors.KernelDensity.
    frequency_max_fit_len: int, default=1000,
        Value to represend the frequency estimator sliding window size for
        X, y and sample weight.
    class_prior : float or array-like, shape (n_classes), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class.
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
        frequency_estimator=None,
        frequency_max_fit_len=1000,
        class_prior=0.0,
        random_state=None,
    ):
        SklearnClassifier.__init__(
            self,
            estimator=estimator,
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        ClassFrequencyEstimator.__init__(
            self,
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.frequency_max_fit_len = frequency_max_fit_len
        self.class_prior = class_prior
        self.frequency_estimator = frequency_estimator

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
        self: KernelFrequencyClassifier,
            The KernelFrequencyClassifier is fitted on the training data.
        """
        self._fit_frequency_estimator(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )
        return super().fit(X, y, sample_weight, **fit_kwargs)

    def partiat_fit(self, X, y, sample_weight=None, **fit_kwargs):
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
        self : KernelFrequencyClassifier,
            The KernelFrequencyClassifier is fitted on the training data.
        """
        self._fit_frequency_estimator(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )
        return super().partial_fit(X, y, sample_weight, **fit_kwargs)

    def _fit_frequency_estimator(
        self, fit_function, X, y, sample_weight=None, **fit_kwargs
    ):
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        check_scalar(
            self.frequency_max_fit_len,
            "m_max",
            int,
            min_val=0,
            min_inclusive=False,
        )

        X, y, sample_weight = ClassFrequencyEstimator._validate_data(
            self,
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

        if not hasattr(self, "frequency_estimator_"):
            if self.frequency_estimator is None:
                self.frequency_estimator_ = KernelDensity()
            else:
                self.frequency_estimator_ = deepcopy(self.frequency_estimator)

        is_lbld = ~np.isnan(y)
        self._label_counts = [
            np.sum(y[is_lbld] == c) for c in range(len(self._le.classes_))
        ]
        try:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)
            if np.sum(is_lbld) == 0:
                raise ValueError("There is no labeled data.")
            elif (
                not has_fit_parameter(
                    self.frequency_estimator_, "sample_weight"
                )
                or sample_weight is None
            ):
                if fit_function == "partial_fit":
                    if not hasattr(self, "X_train_"):
                        self.X_train_ = deque(
                            maxlen=self.frequency_max_fit_len
                        )
                    if not hasattr(self, "y_train_"):
                        self.y_train_ = deque(
                            maxlen=self.frequency_max_fit_len
                        )
                    self.X_train_.extend(X_lbld)
                    self.y_train_.extend(y_lbld)
                    self.frequency_estimator_.fit(
                        X=self.X_train_, y=self.y_train
                    )
                elif fit_function == "fit":
                    self.X_train_ = deque(maxlen=self.frequency_max_fit_len)
                    self.y_train_ = deque(maxlen=self.frequency_max_fit_len)
                    self.X_train_.extend(X_lbld)
                    self.y_train_.extend(y_lbld)
                    self.frequency_estimator_.fit(
                        X=self.X_train_, y=self.y_train_
                    )
            else:
                if fit_function == "partial_fit":
                    if not hasattr(self, "X_train_"):
                        self.X_train_ = deque(
                            maxlen=self.frequency_max_fit_len
                        )
                    if not hasattr(self, "y_train_"):
                        self.y_train_ = deque(
                            maxlen=self.frequency_max_fit_len
                        )
                    if not hasattr(self, "sample_weight_"):
                        self.sample_weight_ = deque(
                            maxlen=self.frequency_max_fit_len
                        )
                    self.X_train_.extend(X_lbld)
                    self.y_train_.extend(y_lbld)
                    self.sample_weight_.extend(sample_weight[is_lbld])
                    self.frequency_estimator_.fit(
                        X=self.X_train_,
                        y=self.y_train_,
                        sample_weight=self.sample_weight_,
                    )
                elif fit_function == "fit":
                    self.X_train_ = deque(maxlen=self.frequency_max_fit_len)
                    self.y_train_ = deque(maxlen=self.frequency_max_fit_len)
                    self.sample_weight_ = deque(
                        maxlen=self.frequency_max_fit_len
                    )
                    self.X_train_.extend(X_lbld)
                    self.y_train_.extend(y_lbld)
                    self.sample_weight_.extend(sample_weight[is_lbld])
                    self.frequency_estimator_.fit(
                        X=self.X_train_,
                        y=self.y_train_,
                        sample_weight=self.sample_weight_,
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

    def predict_freq(self, X):
        """Return class frequency estimates for the input samples 'X'.

        Parameters
        ----------
        X: array-like or shape (n_samples, n_features) or shape
        (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        F: array-like of shape (n_samples, classes)
            The class frequency estimates of the input samples. Classes are
            ordered according to `classes_`.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Predict zeros because of missing training data.
        if self.n_features_in_ is None or not hasattr(self, "X_train_"):
            return np.zeros((len(X), len(self.classes_)))

        frequencys = []
        for x in X:
            frequencys.extend(self._predict_freq(x))
        return np.array(frequencys)

    def _predict_freq(self, X):
        X = np.array(X).reshape([1, -1])
        frequency = np.exp(self.frequency_estimator_.score_samples(X))
        pred_proba = self.estimator_.predict_proba(X)
        return np.array(frequency) * pred_proba + self.class_prior_

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")
