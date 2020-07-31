import numpy as np

from sklearn.base import BaseEstimator


class IgnUnlabeledClassifier(BaseEstimator):
    def __init__(self, estimator: BaseEstimator, unlabeled_class=np.nan):
        # TODO: classes
        self.unlabeled_class = unlabeled_class
        self.estimator = estimator

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
        labeled = self.is_labeled(y)
        self.estimator.fit(X[labeled], y[labeled])

    def predict(self, X, **predict_kwargs):
        """
        Estimator predictions for X. Interface with the predict method of the estimator.
        Args:
            X: The samples to be predicted.
            **predict_kwargs: Keyword arguments to be passed to the predict method of the estimator.
        Returns:
            Estimator predictions for X.
        """
        return self.estimator.predict(X, **predict_kwargs)

    def predict_proba(self, X, **predict_proba_kwargs):
        """
        Class probabilities if the predictor is a classifier. Interface with the predict_proba method of the classifier.
        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the predict_proba method of the classifier.
        Returns:
            Class probabilities for X.
        """
        return self.estimator.predict_proba(X, **predict_proba_kwargs)

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
