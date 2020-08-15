import numpy as np
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.utils.validation import check_random_state, check_is_fitted
from ..utils import MISSING_LABEL, ExtLabelEncoder


class SklearnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, classes=None, missing_label=MISSING_LABEL, random_state=None):
        if not is_classifier(estimator=estimator) or not hasattr(estimator, 'predict_proba'):
            raise TypeError("'{}' must be a classifier implementing 'predict_proba'".format(estimator))
        self.estimator = estimator
        self._le = ExtLabelEncoder(classes=classes, missing_label=missing_label)
        if classes is not None:
            self.classes_ = self._le.classes_
            self.is_fitted_ = False
            self.label_counts_ = np.zeros(len(self.classes_))
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        y_enc = self._le.fit_transform(y)
        is_lbld = ~np.isnan(y_enc)
        try:
            self.estimator.fit(X[is_lbld], y_enc[is_lbld])
            self.is_fitted_ = True
        except Exception as err:
            warnings.warn("'{}' could not be fitted due to: {}".format(self.estimator.__str__(), err), UserWarning)
            if len(self._le.classes_) == 0:
                raise ValueError(
                    "You cannot fit a classifier on empty data, if parameter 'classes' has not been specified.")
            self.is_fitted_ = False
            self.label_counts_ = [np.sum(y_enc[is_lbld] == c) for c in range(len(self._le.classes_))]
        self.classes_ = self._le.classes_
        return self

    def predict(self, X, **predict_kwargs):
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
            if sum(self.label_counts_) == 0:
                return np.ones([len(X), len(self.classes_)]) / len(self.classes_)
            else:
                return np.tile(self.label_counts_ / np.sum(self.label_counts_), [len(X), 1])
