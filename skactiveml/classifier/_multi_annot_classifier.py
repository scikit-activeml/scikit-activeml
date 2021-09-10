"""
Classifier Ensemble for Multiple Annotators
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>

from copy import deepcopy

import numpy as np
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import SkactivemlClassifier
from ..utils import MISSING_LABEL, compute_vote_vectors, is_labeled


class MultiAnnotClassifier(_BaseHeterogeneousEnsemble, SkactivemlClassifier):
    """MultiAnnotClassifier

    This strategy consists of fitting one classifier per annotator.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance.
    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : {scalar, string, np.nan, None}, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with cost_matrix[i,j] indicating cost of predicting class
        classes[j]  for a sample of class classes[i]. Can be only set, if
        classes is not none.
    random_state : int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes)
        Holds the label for each class after fitting.
    cost_matrix_ : array-like, shape (classes, classes)
        Cost matrix with C[i,j] indicating cost of predicting class classes_[j]
        for a sample of class classes_[i].
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    def __init__(self, estimators, voting='hard', classes=None,
                 missing_label=MISSING_LABEL, cost_matrix=None,
                 random_state=None):
        _BaseHeterogeneousEnsemble.__init__(self, estimators=estimators)
        SkactivemlClassifier.__init__(self, classes=classes,
                                      missing_label=missing_label,
                                      cost_matrix=cost_matrix,
                                      random_state=random_state)
        self.voting = voting

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_estimators)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute 'missing_label'.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.

        Returns
        -------
        self: MultiAnnotClassifier,
            The MultiAnnotClassifier is fitted on the training data.
        """
        self._validate_estimators()
        X, y, sample_weight = self._validate_data(
            X=X, y=y, sample_weight=sample_weight
        )
        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting={})"
                             .format(self.voting))
        self._check_n_features(X, reset=True)
        self.estimators_ = deepcopy(self.estimators)

        # Check for empty training data.
        if len(X) == 0:
            return self

        # Fit each estimator.
        for i, est in enumerate(self.estimators_):
            est[1].set_params(missing_label=np.nan)
            if self.classes is None or est[1].classes is None:
                est[1].set_params(classes=np.arange(len(self.classes_)))
            if sample_weight is None:
                est[1].fit(X=X, y=y[:, i])
            else:
                w = sample_weight[:, i] if sample_weight.ndim > 1 else \
                    sample_weight
                est[1].fit(X=X, y=y[:, i], sample_weight=w)
        return self

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or
        shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to classes_.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        if self.n_features_in_ is None:
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)
        elif self.voting == 'hard':
            y_pred = np.array(
                [est.predict(X) for _, est in self.estimators_]).T
            V = compute_vote_vectors(y=y_pred, classes=self.classes_)
            P = V / np.sum(V, axis=1, keepdims=True)
        elif self.voting == 'soft':
            P = np.array(
                [est.predict_proba(X) for _, est in self.estimators_])
            P = np.sum(P, axis=0)
            P /= np.sum(P, axis=1, keepdims=True)
        return P

    def _validate_estimators(self):
        _BaseHeterogeneousEnsemble._validate_estimators(self)
        for name, est in self.estimators:
            if not isinstance(est, SkactivemlClassifier):
                raise TypeError(
                    f"'{est}' is not a 'SkactivemlClassifier'."
                )
            if self.voting == 'soft' and not hasattr(est, 'predict_proba'):
                raise ValueError(
                    f"If 'voting' is soft, each classifier must "
                    f"implement 'predict_proba' method. However, "
                    f"{est} does not do so."
                )
            error_msg = f"{est} of 'estimators' has 'missing_label=" \
                        f"{est.missing_label}' as attribute being unequal " \
                        f"to the given 'missing_label={self.missing_label}' " \
                        f"as parameter."
            try:
                if is_labeled([self.missing_label], est.missing_label)[0]:
                    raise TypeError(error_msg)
            except TypeError:
                raise TypeError(error_msg)
            error_msg = f"{est} of 'estimators' has 'classes={est.classes}' " \
                        f"as attribute being unequal to the given 'classes=" \
                        f"{self.classes}' as parameter."
            classes_none = self.classes is None
            est_classes_none = est.classes is None
            if classes_none and not est_classes_none:
                raise ValueError(error_msg)
            if not classes_none and not est_classes_none and \
                    not np.array_equal(self.classes, est.classes):
                raise ValueError(error_msg)

    def _validate_data(self, X, y, sample_weight):
        X, y, sample_weight = SkactivemlClassifier. \
            _validate_data(self, X=X, y=y, sample_weight=sample_weight)
        error_msg = f"'y' must have shape (n_samples={len(y)}, n_estimators=" \
                    f"{len(self.estimators)}) but has shape {y.shape}."
        if len(X) > 0:
            if y.ndim <= 1 and len(self.estimators) == 1:
                y = y.reshape(len(X), len(self.estimators))
            if y.ndim <= 1 or y.shape[1] != len(self.estimators):
                raise ValueError(error_msg)
        return X, y, sample_weight
