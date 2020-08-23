"""
Wrapper to deal with missing labels and labels from multiple annotators.
"""
import numpy as np
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.utils.validation import check_random_state, check_is_fitted
from ..utils import MISSING_LABEL, ExtLabelEncoder


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    """SklearnClassifier

    Implementation of a wrapper class for scikit-learning classifiers such that missing labels can be handled.

    Parameters
    ----------
    estimator: sklearn.base.ClassifierMixin with 'predict_proba' method
        A scikit-learn classifier that is to deal with missing labels.
    classes: array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    random_state: int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for reproducible results across multiple
        method calls.

    Attributes
    ----------
    classes_: array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    estimator: sklearn.base.ClassifierMixin with 'predict_proba' method
        A scikit-learn classifier that is to deal with missing labels.
    is_fitted_: boolean
        Determines whether the estimator has been fitted or not.
    random_state: int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for reproducible results across multiple
        method calls.
    _le : skactiveml.utils.ExtLabelEncoder
        Encoder for class labels.
    _label_counts: array-like, shape (n_classes)
        Number of observed labels per class.
    """

    def __init__(self, estimator, classes=None, missing_label=MISSING_LABEL, random_state=None):
        if not is_classifier(estimator=estimator) or not hasattr(estimator, 'predict_proba'):
            raise TypeError("'{}' must be a classifier implementing 'predict_proba'".format(estimator))
        self.estimator = estimator
        self._le = ExtLabelEncoder(classes=classes, missing_label=missing_label)
        self.missing_label = self._le.missing_label
        if classes is not None:
            self.classes_ = self._le.classes_
            self.is_fitted_ = False
            self._label_counts = np.zeros(len(self.classes_))
        self.random_state = check_random_state(random_state)

    def fit(self, X, y, **fit_kwargs):
        """
        Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where missing labels are
            represented the attribute 'missing_label'.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: SklearnClassifier,
            The SklearnClassifier is fitted on the training data.
        """
        y_enc = self._le.fit_transform(y)
        is_lbld = ~np.isnan(y_enc)
        try:
            self.estimator.fit(X[is_lbld], y_enc[is_lbld], **fit_kwargs)
            self.is_fitted_ = True
        except Exception as err:
            warnings.warn("'{}' could not be fitted due to: {}".format(self.estimator.__str__(), err), UserWarning)
            if len(self._le.classes_) == 0:
                raise ValueError(
                    "You cannot fit a classifier on empty data, if parameter 'classes' has not been specified.")
            self.is_fitted_ = False
            self._label_counts = [np.sum(y_enc[is_lbld] == c) for c in range(len(self._le.classes_))]
        self.classes_ = self._le.classes_
        return self

    def predict(self, X, **predict_kwargs):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the 'predict' method of the 'estimator'.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self, attributes=['is_fitted_'])
        if self.is_fitted_:
            return self._le.inverse_transform(self.estimator.predict(X, **predict_kwargs))
        else:
            warnings.warn("'{}' not fitted: Return default predictions".format(self.estimator.__str__()), UserWarning)
            p = self.predict_proba([X[0]]).ravel()
            return self.random_state.choice(self.classes_, len(X), replace=True, p=p)

    def predict_proba(self, X, **predict_proba_kwargs):
        """Return probability estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the 'predict_proba' method of the 'estimator'.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        check_is_fitted(self, attributes=['is_fitted_'])
        if self.is_fitted_:
            P = self.estimator.predict_proba(X, **predict_proba_kwargs)
            if len(self.estimator.classes_) != len(self.classes_):
                P_ext = np.zeros((len(X), len(self.classes_)))
                class_indices = np.asarray(self.estimator.classes_, dtype=int)
                P_ext[:, class_indices] = P
                P = P_ext
            return P
        else:
            if sum(self._label_counts) == 0:
                return np.ones([len(X), len(self.classes_)]) / len(self.classes_)
            else:
                return np.tile(self._label_counts / np.sum(self._label_counts), [len(X), 1])
