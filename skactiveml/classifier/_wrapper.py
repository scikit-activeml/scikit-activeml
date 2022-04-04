"""
Wrapper for scikit-learn classifiers to deal with missing labels and labels
from multiple annotators.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>


import warnings
from copy import copy, deepcopy
from collections import deque
from xmlrpc.client import boolean

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, is_classifier
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    has_fit_parameter,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.neighbors import KernelDensity
from ..base import SkactivemlClassifier, ClassFrequencyEstimator
from ..utils import (
    rand_argmin,
    MISSING_LABEL,
    check_scalar,
    call_func,
    check_type,
    is_labeled,
    check_cost_matrix,
    check_classifier_params,
    check_random_state,
    ExtLabelEncoder,
    check_class_prior,
)


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
        # count labels per class
        is_lbld = ~np.isnan(y)
        self._label_counts = [
            np.sum(y[is_lbld] == c) for c in range(len(self._le.classes_))
        ]
        try:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)

            if np.sum(is_lbld) == 0:
                if (
                    hasattr(self, "is_fitted_")
                    and self.is_fitted_ is True
                    and fit_function == "partial_fit"
                ):
                    return self
                else:
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


class KernelFrequencyClassifier(ClassFrequencyEstimator):
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
        X, y and sample weight. If 'None' the windows is unrestricted in size.
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
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.estimator = estimator
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
        return self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    def predict(self, X):
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
        return self.estimator_.predict(X)

    def predict_proba(self, X):
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
        proba = self.estimator_.predict_proba(X)
        return proba

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

        if hasattr(self.frequency_estimator_, "predict_freq"):
            return self.frequency_estimator_.predict_freq(X=X)
        # Predict zeros because of missing training data.
        if not self.is_fitted_:
            return np.zeros((len(X), len(self.classes_)))
        F = []
        for x in X:
            F.extend(self._predict_freq(x))
        return np.array(F)

    def _predict_freq(self, X):
        X = np.array(X).reshape([1, -1])
        frequency = np.exp(self.frequency_estimator_.score_samples(X))
        pred_proba = self.estimator_.predict_proba(X)
        return np.array(frequency) * pred_proba + self.class_prior_

    def _fit(self, fit_function, X, y, sample_weight=None, **fit_kwargs):
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

        # Check whether estimator is a valid classifier.
        if not isinstance(
            self.estimator, SklearnClassifier
        ) and not is_classifier(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier or scikit-learn "
                "classifier.".format(self.estimator)
            )

        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

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
        sample_weight_train = sample_weight
        if fit_function == "fit":
            self.estimator_.fit(
                X=X, y=y, sample_weight=sample_weight_train, **fit_kwargs
            )
        elif fit_function == "partial_fit":
            self.estimator_.partial_fit(
                X=X, y=y, sample_weight=sample_weight_train, **fit_kwargs
            )
        if hasattr(self.estimator_, "is_fitted_"):
            self.is_fitted_ = self.estimator_.is_fitted_

        return self

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

        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

        if not hasattr(self, "frequency_estimator_"):
            if self.frequency_estimator is None:
                self.frequency_estimator_ = SubSampleEstimator(
                    KernelDensity(),
                    missing_label=self.missing_label,
                    classes=self.classes,
                    max_fit_len=self.frequency_max_fit_len,
                )
            else:
                self.frequency_estimator_ = deepcopy(self.frequency_estimator)
        X_train = X
        y_train = y
        sample_weight_train = sample_weight

        if fit_function == "partial_fit":
            call_func(
                self.frequency_estimator_.partial_fit,
                X=X_train,
                y=y_train,
                sample_weight=sample_weight_train,
            )
        elif fit_function == "fit":
            call_func(
                self.frequency_estimator_.fit,
                X=X_train,
                y=y_train,
                sample_weight=sample_weight_train,
            )

    

    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        check_y_dict=None,
        y_ensure_1d=True,
    ):
        if check_y_dict is None:
            check_y_dict = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "ensure_2d": False,
                "force_all_finite": False,
                "dtype": None,
            }

        # Check common classifier parameters.
        check_classifier_params(
            self.classes, self.missing_label, self.cost_matrix
        )

        # Store and check random state.
        self.random_state_ = check_random_state(self.random_state)

        # Create label encoder.
        self._le = ExtLabelEncoder(
            classes=self.classes, missing_label=self.missing_label
        )

        # Check input parameters.
        y = check_array(y, **check_y_dict)
        if len(y) > 0:
            y_le = column_or_1d(y) if y_ensure_1d else y
            y_le = self._le.fit_transform(y_le)
            is_lbdl = is_labeled(y_le)

            if len(y_le[is_lbdl]) > 0:
                check_classification_targets(y_le[is_lbdl])
            if len(self._le.classes_) == 0:
                raise ValueError(
                    "No class label is known because 'y' contains no actual "
                    "class labels and 'classes' is not defined. Change at "
                    "least on of both to overcome this error."
                )
            self._label_counts = [
                np.sum(y_le[is_lbdl] == c)
                for c in range(len(self._le.classes_))
            ]
        else:
            self._le.fit_transform(self.classes)
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)

        # Update detected classes.
        self.classes_ = self._le.classes_

        # Check classes.
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f"`y` has the shape {y.shape} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"identical shapes."
                )

        # Update cost matrix.
        self.cost_matrix_ = (
            1 - np.eye(len(self.classes_))
            if self.cost_matrix is None
            else self.cost_matrix
        )
        self.cost_matrix_ = check_cost_matrix(
            self.cost_matrix_, len(self.classes_)
        )
        if self.classes is not None:
            class_indices = np.argsort(self.classes)
            self.cost_matrix_ = self.cost_matrix_[class_indices]
            self.cost_matrix_ = self.cost_matrix_[:, class_indices]

        # Check class prior.
        self.class_prior_ = check_class_prior(
            self.class_prior, len(self.classes_)
        )

        return X, y, sample_weight

    def __getattr__(self, item):
        if "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")


class SubSampleEstimator(SkactivemlClassifier, MetaEstimatorMixin):
    """SubSampleEstimator

    Implementation of a wrapper class for scikit-learn classifiers such that
    missing labels can be handled. Therefor, samples with missing labels are
    filtered. Furthermore, saves X, y and sample_weight in a
    sliding window, enabeling the simulation of a partial fit function for any
    classifier.

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
    max_fit_len: int, default=None,
        Value to represend the estimator sliding window size for X, y and
        sample weight. If 'None' the windows is unrestricted in size.
    handle_window: str, default='last'
        The decision on how to handel the sliding window when the max_fit_len
        is reached. 
        'last': First in First out
        'random' : randomly sort previous data and add the new indices
    only_labled: bool, default=False
        decides if only labled data should be saved of all data.
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
        max_fit_len=None,
        handle_window="last",
        only_labled=False,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.estimator = estimator
        self.only_labled = only_labled
        self.max_fit_len = max_fit_len
        self.handle_window = handle_window

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as class labels. Resets
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
        self: SubSampleEstimator,
            The SubSampleEstimator is fitted on the training data.
        """
        self._handle_window("fit", X, y, sample_weight)
        return self._fit(
            "fit",
            X=self.X_train_,
            y=self.y_train_,
            sample_weight=self.sample_weight_,
            **fit_kwargs,
        )

    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as class
        labels. If 'base_estimator' has no partial_fit function uses fit with 
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
        self : SubSampleEstimator,
            The SubSampleEstimator is fitted on the training data.
        """
        self._handle_window("partial_fit", X, y, sample_weight)

        if hasattr(self.estimator, "partial_fit"):
            return self._fit(
                "partial_fit",
                X=X,
                y=y,
                sample_weight=sample_weight,
                **fit_kwargs,
            )
        else:
            return self._fit(
                "fit",
                X=self.X_train_,
                y=self.y_train_,
                sample_weight=self.sample_weight_,
                **fit_kwargs,
            )

    def _handle_window(self, fit_func, X, y, sample_weight=None):
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

        if not hasattr(self, "X_train_"):
            self.X_train_ = deque(maxlen=self.max_fit_len)
        if not hasattr(self, "y_train_"):
            self.y_train_ = deque(maxlen=self.max_fit_len)
        if not hasattr(self, "sample_weight_"):
            self.sample_weight_ = deque(maxlen=self.max_fit_len)
        # reset the window if fit is called otherwise extend the window with
        # the given data
        if fit_func == "fit":
            self.X_train_ = deque(maxlen=self.max_fit_len)
            self.y_train_ = deque(maxlen=self.max_fit_len)
            self.sample_weight_ = deque(maxlen=self.max_fit_len)
        elif fit_func == "partial_fit":
            if (
                self.max_fit_len is not None
                and len(self.X_train_) + len(X) >= self.max_fit_len
            ):
                if self.handle_window == "random":
                    index = np.random.choice(
                        len(self.X_train_),
                        len(self.X_train_) - len(X),
                        replace=False,
                    )
                    # since old data will get removed when max_fit_len is
                    # reached extend the randomized data to the window.
                    self.X_train_.extend(np.array(self.X_train_)[index])
                    self.y_train_.extend(np.array(self.y_train_)[index])
                    self.sample_weight_.extend(
                        np.array(self.sample_weight_)[index]
                    )
        self.X_train_.extend(X)
        self.y_train_.extend(y)
        if sample_weight is not None:
            self.sample_weight_.extend(sample_weight)
        else:
            self.sample_weight_ = None

    def _fit(self, fit_function, X, y, sample_weight=None, **fit_kwargs):
        # Check input parameters.
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        # Check whether estimator is a valid classifier.
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(
                "'{}' must be a BaseEstimator "
                "classifier.".format(self.estimator)
            )

        X, y, sample_weight, is_lbld = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
            return_is_lbld=True,
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

        try:
            if self.only_labled:
                X = X[is_lbld]
                y = y[is_lbld]
                if sample_weight is not None:
                    sample_weight = sample_weight[is_lbld]
                else:
                    sample_weight = None
                if np.sum(is_lbld) == 0:
                    if (
                        hasattr(self, "is_fitted_")
                        and self.is_fitted_ is True
                        and fit_function == "partial_fit"
                    ):
                        return self
                    else:
                        raise ValueError("There is no labeled data.")
            if (
                not isinstance(self.estimator_, SkactivemlClassifier)
                and np.sum(is_lbld) == 0
            ):
                self.estimator_ = SklearnClassifier(
                    self.estimator_,
                    classes=self.classes,
                    missing_label=self.missing_label,
                    cost_matrix=self.cost_matrix,
                    random_state=self.random_state,
                )
                warnings.warn(
                    "The 'base_estimator' is not an SkactivemlClassifier and fitted with no Labeled data, Therfore it will be wrapped into an SklearnClassifier"
                )
            if (
                not has_fit_parameter(self.estimator, "sample_weight")
                or sample_weight is None
            ):
                if fit_function == "partial_fit":
                    classes = self._le.transform(self.classes_)
                    # classes is called 2 times if estimator is a
                    # SklearnClassifier
                    if isinstance(self.estimator_, SklearnClassifier):
                        self.estimator_.classes_ = classes
                        self.estimator_.partial_fit(X=X, y=y, **fit_kwargs)
                    else:
                        self.estimator_.partial_fit(
                            X=X, y=y, classes=classes, **fit_kwargs
                        )
                elif fit_function == "fit":
                    self.estimator_.fit(X=X, y=y, **fit_kwargs)
            else:
                if fit_function == "partial_fit":
                    classes = self._le.transform(self.classes_)
                    self.estimator_.partial_fit(
                        X=X,
                        y=y,
                        classes=classes,
                        sample_weight=sample_weight,
                        **fit_kwargs,
                    )
                elif fit_function == "fit":
                    self.estimator_.fit(
                        X=X, y=y, sample_weight=sample_weight, **fit_kwargs,
                    )
            self.is_fitted_ = True
        except Exception as e:
            self.is_fitted_ = False
            if hasattr(self, "_label_counts"):
                warnings.warn(
                    "The 'base_estimator' could not be fitted because of"
                    " '{}'. Therefore, the class labels of the samples "
                    "are counted and will be used to make predictions. "
                    "The class label distribution is `_label_counts={}`.".format(
                        e, self._label_counts
                    )
                )
            else:
                warnings.warn(
                    "The 'base_estimator' could not be fitted because of"
                    " '{}'.".format(e)
                )
        return self


    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        check_y_dict=None,
        y_ensure_1d=True,
        return_is_lbld=False,
    ):

        # create new y array to check classes
        if hasattr(self, "y_train_"):
            y_new = np.concatenate([self.y_train_, y])
        else:
            y_new = y

        if self.max_fit_len is not None:
            check_scalar(self.max_fit_len, "max_fit_len", int, min_val=0)
        check_type(self.only_labled, "only_labled", bool)
        check_type(self.handle_window, "handle_window", str)
        if check_y_dict is None:
            check_y_dict = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "ensure_2d": False,
                "force_all_finite": False,
                "dtype": None,
            }

        # Check common classifier parameters.
        check_classifier_params(
            self.classes, self.missing_label, self.cost_matrix
        )

        # Store and check random state.
        self.random_state_ = check_random_state(self.random_state)

        # Create label encoder.
        self._le = ExtLabelEncoder(
            classes=self.classes, missing_label=self.missing_label
        )
        # Check input parameters.
        y = check_array(y, **check_y_dict)
        if len(y) > 0:
            y_le = column_or_1d(y_new) if y_ensure_1d else y_new
            y_le = self._le.fit_transform(y_new)
            is_lbdl = is_labeled(y_le)

            if len(y_le[is_lbdl]) > 0:
                check_classification_targets(y_le[is_lbdl])
            if len(self._le.classes_) == 0:
                raise ValueError(
                    "No class label is known because 'y' contains no actual "
                    "class labels and 'classes' is not defined. Change at "
                    "least one of both to overcome this error."
                )
            self._label_counts = [
                np.sum(y_le[is_lbdl] == c)
                for c in range(len(self._le.classes_))
            ]
        else:
            y_new = check_array(y_new, **check_y_dict)
            is_lbdl = np.ones(y_new.shape, dtype=bool)
            self._le.fit_transform(self.classes)
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)

        # Update detected classes.
        self.classes_ = self._le.classes_

        # Check classes.
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f"`y` has the shape {y.shape} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"identical shapes."
                )

        # Update cost matrix.
        self.cost_matrix_ = (
            1 - np.eye(len(self.classes_))
            if self.cost_matrix is None
            else self.cost_matrix
        )
        self.cost_matrix_ = check_cost_matrix(
            self.cost_matrix_, len(self.classes_)
        )
        if self.classes is not None:
            class_indices = np.argsort(self.classes)
            self.cost_matrix_ = self.cost_matrix_[class_indices]
            self.cost_matrix_ = self.cost_matrix_[:, class_indices]

        if return_is_lbld:
            is_lbdl = is_lbdl[len(self.y_train_) :]
            return X, y, sample_weight, is_lbdl
        else:
            return X, y, sample_weight

    def predict(self, X):
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
        return self.estimator_.predict(X)

    def predict_proba(self, X):
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
        proba = self.estimator_.predict_proba(X)
        return proba

    def __getattr__(self, item):
        if "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")