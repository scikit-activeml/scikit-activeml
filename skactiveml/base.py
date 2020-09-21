import numpy as np

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state, check_array, \
    check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score
from skactiveml.utils import MISSING_LABEL, check_classifier_params, \
    rand_argmin, ExtLabelEncoder, check_cost_matrix, is_labeled


class QueryStrategy(ABC, BaseEstimator):

    def __init__(self, random_state=None):
        # set RS
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def query(self, *args, **kwargs):
        return NotImplemented


class PoolBasedQueryStrategy(QueryStrategy):

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    @abstractmethod
    def query(self, X_cand, *args, return_utilities=False, **kwargs):
        return NotImplemented


class StreamBasedQueryStrategy(QueryStrategy):
    """Base class for all stream-based active learning query strategies in
       scikit-activeml.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget_manager, random_state=None):
        super().__init__(random_state=random_state)
        self.budget_manager = budget_manager

    @abstractmethod
    def query(self, X_cand, *args, return_utilities=False, simulate=False,
              **kwargs):
        """Ask the query strategy which instances in X_cand to acquire.

        The query startegy determines the most useful instances in X_cand,
        which can be acquired within the budgeting constraint specified by the
        budget_manager.
        Please note that, when the decisions from this function
        may differ from the final sampling, simulate=True can set, so that the
        query strategy can be updated later with update(...) with the final
        sampling. This is especially helpful, when developing wrapper query
        strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        simulate : bool, optional
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Note, that this is propagated
            to the budget_manager, as well. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        return NotImplemented

    @abstractmethod
    def update(self, X_cand, sampled, *args, **kwargs):
        """Update the query strategy with the decisions taken.

        This function should be used in conjunction with the query function,
        when the instances sampled from query(...) may differ from the
        instances sampled in the end. In this case use query(...) with
        simulate=true and provide the final decisions via update(...).
        This is especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : StreamBasedQueryStrategy
            The StreamBasedQueryStrategy returns itself, after it is updated.
        """
        return NotImplemented


class SkactivemlClassifier(BaseEstimator, ClassifierMixin, ABC):

    @abstractmethod
    def __init__(self, classes, missing_label=MISSING_LABEL, cost_matrix=None,
                 random_state=None):
        # Check common classifier parameters.
        self.classes, self.missing_label, self.cost_matrix = \
            check_classifier_params(classes, missing_label, cost_matrix)

        # Store and check random state.
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute 'missing_label'.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.

        Returns
        -------
        self: SkactivemlClassifier,
            The SkactivemlClassifier is fitted on the training data.
        """
        return NotImplemented

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y = self._le.transform(y)
        y_pred = self._le.transform(self.predict(X))
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def _validate_input(self, X, y, sample_weight):
        # Check input parameters.
        X = check_array(X)
        y = np.array(y)
        check_consistent_length(X, y)
        is_lbdl = is_labeled(y, self.missing_label)
        if len(y[is_lbdl]) > 0:
            y_type = type_of_target(y[is_lbdl])
            if y_type not in ['binary', 'multiclass', 'multiclass-multioutput',
                              'multilabel-indicator', 'multilabel-sequences',
                              'unknown']:
                raise ValueError("Unknown label type: %r" % y_type)
        self._le = ExtLabelEncoder(classes=self.classes,
                                   missing_label=self.missing_label)
        y = self._le.fit_transform(y)
        if len(self._le.classes_) == 0:
            raise ValueError("No class label is known because 'y' contains no "
                             "actual class labels and 'classes' is not "
                             "defined. Change at least on of both to overcome "
                             "this error.")
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(y, sample_weight)
            if y.ndim > 1 and y.shape[1] > 1 or \
                    sample_weight.ndim > 1 and sample_weight.shape[1] > 1:
                check_consistent_length(y.T, sample_weight.T)

        # Update detected classes.
        self.classes_ = self._le.classes_

        # Update cost matrix.
        self.cost_matrix_ = 1 - np.eye(len(self.classes_)) \
            if self.cost_matrix is None else self.cost_matrix
        self.cost_matrix_ = check_cost_matrix(self.cost_matrix_,
                                              len(self.classes_))
        if self.classes is not None:
            class_indices = np.argsort(self.classes)
            self.cost_matrix_ = self.cost_matrix_[class_indices]
            self.cost_matrix_ = self.cost_matrix_[:, class_indices]
        return X, y, sample_weight

    def _more_tags(self):
        return {'multioutput_only': True}


class ClassFrequencyEstimator(SkactivemlClassifier):

    @abstractmethod
    def predict_freq(self, X):
        """Return class frequency estimates for the input data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features) or shape
        (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        F: array-like, shape (n_samples, classes)
            The class frequency estimates of the input samples. Classes are
            ordered according to classes_.
        """
        return NotImplemented

    def predict_proba(self, X):
        """Return probability estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or
        shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            according to classes_.
        """
        # Normalize probabilities of each sample.
        P = self.predict_freq(X)
        normalizer = np.sum(P, axis=1)
        P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
        P[normalizer == 0, :] = [1 / len(self.classes_)] * len(self.classes_)
        return P

    def predict(self, X):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features) or
        shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        P = self.predict_proba(X)
        costs = np.dot(P, self.cost_matrix_)
        y_pred = rand_argmin(costs, random_state=self.random_state, axis=1)
        y_pred = self._le.inverse_transform(y_pred)
        y_pred = np.asarray(y_pred, dtype=self.classes_.dtype)
        return y_pred
