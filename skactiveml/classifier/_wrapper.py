"""
Wrapper for scikit-learn classifiers to deal with missing labels and labels
from multiple annotators.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>


import warnings
from copy import deepcopy
from collections import deque

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
from . import ParzenWindowClassifier
from sklearn.utils.multiclass import check_classification_targets
from ..base import SkactivemlClassifier, ClassFrequencyEstimator
from ..utils import (
    rand_argmin,
    MISSING_LABEL,
    check_scalar,
    check_type,
    is_labeled,
    check_cost_matrix,
    check_classifier_params,
    check_random_state,
    ExtLabelEncoder,
    check_class_prior,
    check_equal_missing_label,
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
        is_lbld = is_labeled(y, missing_label=-1)
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

    Implementation of a wrapper class for skactiveml classifiers such that
    `predict_freq` can be used by calculating the frequencies using other
    models (e.g., KernelDensity from scikit-learn or ParzenWindowClassifier).

    Parameters
    ----------
    estimator : sklearn.base.SkactivemlClassifier
        The `SkactivemlClassifier` where the `predict_freq` method should be
        added.
    class_frequency_estimator : sklearn.base.ClassFrequencyEstimator,
        default=None
        The `ClassFrequencyEstimator` from which the `predict_freq` method
        should utilized.
    use_only_marginal_frequencies : boolean, default=True
        If True, the estimated class frequencies from the
        `class_frequency_estimator` are marginalized over all classes and
        multiplied with the predicted class probabilities from the `estimator`.
        If False, the frequencies provided by the `class_frequency_estimator`
        are returned as is.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit. Is ignored if class_frequency_estimator is not None.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none. Is ignored if class_frequency_estimator is not
        None.
    class_prior : float or array-like, shape (n_classes), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class. Is ignored if
        class_frequency_estimator is not None.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self,
        estimator,
        class_frequency_estimator=None,
        use_only_marginal_frequencies=True,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        class_prior=0.0,
        random_state=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            class_prior=class_prior,
            random_state=random_state,
        )
        self.class_frequency_estimator = class_frequency_estimator
        self.use_only_marginal_frequencies = use_only_marginal_frequencies
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
        self: KernelFrequencyClassifier,
            The KernelFrequencyClassifier is fitted on the training data.
        """
        self._fit(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )
        self._fit_frequency_estimator(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )
        return self.estimator_

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
        self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )
        self._fit_frequency_estimator(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )
        return self.estimator_

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
        # Predict zeros because of missing training data.
        # Only required because of the use of SubsampleEstimator
        if not self.is_fitted_:
            return np.zeros((len(X), len(self.classes_)))
        frequencies = self.class_frequency_estimator_.predict_freq(X)
        if not self.use_only_marginal_frequencies:
            return frequencies
        marginal_frequencies = np.sum(frequencies, axis=1, keepdims=True)
        pred_proba = self.estimator_.predict_proba(X)
        return marginal_frequencies * pred_proba

    def _fit(self, fit_function, X, y, sample_weight=None, **fit_kwargs):
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        # Check whether estimator is a valid classifier.
        if not isinstance(self.estimator, SkactivemlClassifier):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier"
                "classifier.".format(self.estimator)
            )

        if self.class_frequency_estimator is not None and not isinstance(
            self.class_frequency_estimator, SkactivemlClassifier
        ):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier"
                "classifier.".format(self.class_frequency_estimator)
            )

        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
        )

        if not hasattr(self, "class_frequency_estimator_"):
            if self.class_frequency_estimator is None:
                is_labeled(y, missing_label=self.missing_label)
                self.class_frequency_estimator_ = SubsampleEstimator(
                    ParzenWindowClassifier(
                        missing_label=self.missing_label,
                        classes=self.classes,
                        random_state=self.random_state_.randint(2 ** 31 - 1),
                        class_prior=self.class_prior,
                        cost_matrix=self.cost_matrix,
                    ),
                    only_labled=True,
                    classes=self.classes,
                    missing_label=self.missing_label,
                    random_state=self.random_state_.randint(2 ** 31 - 1),
                )
            else:
                self.class_frequency_estimator_ = deepcopy(
                    self.class_frequency_estimator
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
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_

        if hasattr(self.estimator_, "_le"):
            self._le = self.estimator_._le

        return self

    def _fit_frequency_estimator(
        self, fit_function, X, y, sample_weight=None, **fit_kwargs
    ):
        check_equal_missing_label(
            self.class_frequency_estimator_.missing_label,
            self.estimator.missing_label,
        )

        # Check class prior.
        self.class_prior_ = check_class_prior(
            self.class_prior, len(self.classes_)
        )
        if fit_function == "partial_fit":
            fit_callback = self.class_frequency_estimator_.partial_fit
        elif fit_function == "fit":
            fit_callback = self.class_frequency_estimator_.fit
        return fit_callback(X, y, sample_weight)

    def _validate_data(
        self, X, y, sample_weight=None, check_X_dict=None, check_y_dict=None,
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

        # Check input parameters.
        y = check_array(y, **check_y_dict)
        check_type(
            self.use_only_marginal_frequencies,
            "use_only_marginal_frequencies",
            bool,
        )
        if len(y) == 0:
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)

        # # Update detected classes.

        # Check classes.
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f"`y` has the shape {y.shape} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"identical shapes."
                )

        return X, y, sample_weight

    def __getattr__(self, item):
        if "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")


class SubsampleEstimator(SkactivemlClassifier, MetaEstimatorMixin):
    """SubsampleEstimator

    Implementation of a wrapper class for SkactivemlClassifier such that the
    number of training samples can be limited. Furthermore, saves X, y and
    sample_weight, enabling the use of a partial fit for any classifier. The
    class offers two updating strategies, i.e., replacing the oldes samples
    (sliding window) or replacing samples randomly.

    Parameters
    ----------
    estimator : sklearn.base.SkactivemlClassifier
        The wrapped classifier.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    subsample_size: int, default=None,
        Value to represent the estimator sliding window size for X, y and
        sample weight. If 'None' the windows is unrestricted in size.
    replacement_method: str, default='last'
        Defines how old instances are replaced.
        'last': First in First out
        'random' : replacing old samples randomly
    only_labled: bool, default=False
        If True, unlabled samples are discarded.
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
        random_state=None,
        subsample_size=None,
        replacement_method="last",
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
        self.subsample_size = subsample_size
        self.replacement_method = replacement_method

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
        self: SubsampleEstimator,
            The SubsampleEstimator is fitted on the training data.
        """
        self._replacement_method("fit", X, y, sample_weight)
        return self._fit(
            "fit",
            X=self.X_train_,
            y=self.y_train_,
            sample_weight=self.sample_weight_,
            **fit_kwargs,
        )

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
        self : SubsampleEstimator,
            The SubsampleEstimator is fitted on the training data.
        """
        self._replacement_method("partial_fit", X, y, sample_weight)

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

    def _replacement_method(self, fit_func, X, y, sample_weight=None):
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
            self.X_train_ = deque(maxlen=self.subsample_size)
        if not hasattr(self, "y_train_"):
            self.y_train_ = deque(maxlen=self.subsample_size)
        if not hasattr(self, "sample_weight_"):
            self.sample_weight_ = deque(maxlen=self.subsample_size)
        # reset the window if fit is called otherwise extend the window with
        # the given data
        if fit_func == "fit":
            self.X_train_ = deque(maxlen=self.subsample_size)
            self.y_train_ = deque(maxlen=self.subsample_size)
            self.sample_weight_ = deque(maxlen=self.subsample_size)
        elif fit_func == "partial_fit":
            if (
                self.subsample_size is not None
                and len(self.X_train_) + len(X) >= self.subsample_size
            ):
                if self.replacement_method == "random":
                    index = np.random.choice(
                        len(self.X_train_),
                        len(self.X_train_) - len(X),
                        replace=False,
                    )
                    # since old data will get removed when subsample_size is
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

        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
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
                is_lbld = is_labeled(y, self.missing_label)
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
    ):

        # create new y array to check classes
        if hasattr(self, "y_train_"):
            y_new = np.concatenate([self.y_train_, y])
        else:
            y_new = y

        if self.subsample_size is not None:
            check_scalar(
                self.subsample_size,
                "subsample_size",
                int,
                min_val=0,
                min_inclusive=False,
            )
        check_type(self.only_labled, "only_labled", bool)
        check_type(self.replacement_method, "replacement_method", str)
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

        if isinstance(self.estimator, SkactivemlClassifier):
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
            if self.only_labled:
                check_equal_missing_label(
                    self.missing_label, self.estimator.missing_label,
                )
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

        # Create label encoder.
        self._le = ExtLabelEncoder(
            classes=self.classes, missing_label=self.missing_label
        )
        # Check input parameters.
        y = check_array(y, **check_y_dict)
        if len(y) > 0:
            y_le = column_or_1d(y_new) if y_ensure_1d else y_new
            y_le = self._le.fit_transform(y_le)
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
