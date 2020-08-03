import numpy as np
import warnings

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

class IgnUnlabeledWrapper(BaseEstimator):
    # TODO How to ensure that the estimator uses the provided labels as label encoder
    def __init__(self, estimator: BaseEstimator, classes=None, unlabeled_class=np.nan):
        # TODO: classes
        self.unlabeled_class = unlabeled_class
        self.estimator = estimator
        self._is_fitted = False

        if classes is not None:
            self.classes_ = classes
            self.update_classes_ = False
        else:
            self.classes_ = np.array([0, 1])
            self.update_classes_ = True

        if self.unlabeled_class is None:
            self._is_unl = lambda elem: elem is self.unlabeled_class
        elif np.isnan(self.unlabeled_class):
            self._is_unl = lambda elem: np.isnan(elem)
        else:
            self._is_unl = lambda elem: elem == self.unlabeled_class

    def is_unlabeled(self, y):
        return np.fromiter((self._is_unl(x) for x in y), 'bool')

    def is_labeled(self, y):
        return np.logical_not(self.is_unlabeled(y))

    def fit(self, X, y):
        if self.update_classes_:
            le = LabelEncoder().fit(y[self.is_labeled(y)])
            self.classes_ = le.classes_
            if len(self.classes_) == 0:
                self.classes_ = np.array([0,1])
            elif len(self.classes_) == 1:
                self.classes_ = np.union1d(self.classes_, np.array([0,1]))[:2]

        labeled = self.is_labeled(y)
        try:
            self.estimator.fit(X[labeled], y[labeled])
            self._is_fitted = True
        except Exception as err:
            self._is_fitted = False
            self._label_counts = [sum(y[self.is_labeled(y)] == c) for c in self.classes_]
            warnings.warn("Classifier '{}' could not be fitted due to: {}".format(self.estimator.__str__(), err), UserWarning)


    def predict(self, X, **predict_kwargs):
        """
        Estimator predictions for X. Interface with the predict method of the estimator.
        Args:
            X: The samples to be predicted.
            **predict_kwargs: Keyword arguments to be passed to the predict method of the estimator.
        Returns:
            Estimator predictions for X.
        """
        if self._is_fitted:
            return self.estimator.predict(X, **predict_kwargs)
        else:
            warnings.warn("Classifier '{}' not fitted: Return default predictions".format(self.estimator.__str__()),
                          UserWarning)
            return np.random.choice(self.classes_, len(X), replace=True)
            #TODO: randomstate?

    def predict_proba(self, X, **predict_proba_kwargs):
        """
        Class probabilities if the predictor is a classifier. Interface with the predict_proba method of the classifier.
        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the predict_proba method of the classifier.
        Returns:
            Class probabilities for X.
        """
        if self._is_fitted:
            return self.estimator.predict_proba(X, **predict_proba_kwargs)
        else:
            if sum(self._label_counts) == 0:
                return np.ones([len(X), len(self.classes_)]) / len(self.classes_)
            else:
                return np.tile(self._label_counts / np.sum(self._label_counts), [len(X), 1])

    def score(self, X, y, **score_kwargs):
        """
        Interface for the score method of the predictor.
        Args:
            X: The samples for which prediction accuracy is to be calculated.
            y: Ground truth labels for X.
            **score_kwargs: Keyword arguments to be passed to the .score() method of the predictor.
        Returns:
            The score of the predictor.
        """
        return self.estimator.score(X, y, **score_kwargs)
