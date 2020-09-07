import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils.validation import check_random_state, check_array, \
    check_is_fitted, NotFittedError, check_consistent_length
from scipy.spatial.distance import cdist
from ..utils import check_cost_matrix, ExtLabelEncoder, MISSING_LABEL, \
    compute_vote_vectors, rand_argmin


class CMM(BaseEstimator, ClassifierMixin):
    """CMM

    Classifier mixture model (CMM) is a generative classifier based on a
    Gaussian mixture model (GMM).

    Parameters
    ----------
    mixture_model : GaussianMixture or BayesianGaussianMixture or None,
    default=BayesianMixtureModel(n_components=10)
        Bayesian Gaussian Mixture model that is trained with unsupervised
        algorithm on train data. If the initial mixture model is not fitted, it
         will be refitted in each call of the 'fit' method.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label : scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    random_state : int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    mixture_model : sklearn.mixture.GaussianMixture | BayesianGaussianMixture
    | None, default=BayesianMixtureModel(n_components=10)
        Bayesian Gaussian Mixture model that is trained with unsupervised
        algorithm on train data. If the initial mixture model is not fitted, it
         will be refitted in each call of the 'fit' method.
    classes_ : array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label : scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    random_state : int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    F_components_ : numpy.ndarray, shape (n_components, n_classes)
        F[j,c] is a proxy for the number of sample of class c belonging to
        component j.
    _le : skactiveml.utils.ExtLabelEncoder
        Encoder for class labels.
    """

    def __init__(self, mixture_model=None, classes=None,
                 missing_label=MISSING_LABEL, cost_matrix=None,
                 random_state=None):
        # Check mixture model.
        self.mixture_model = BayesianGaussianMixture(
            n_components=10) if mixture_model is None else mixture_model
        self._provided_mixture = mixture_model is not None
        if not isinstance(self.mixture_model,
                          (GaussianMixture, BayesianGaussianMixture)):
            raise TypeError(
                "'mixture_model' is of the type '{}' but must be of the type "
                "'sklearn.mixture.GaussianMixture' or "
                "'sklearn.mixture.BayesianGaussianMixture'.".format(
                    type(self.mixture_model)))
        self._refit = False
        try:
            check_is_fitted(self.mixture_model)
        except NotFittedError:
            self._refit = True

        # Setup label encoder.
        self.classes = classes
        self._le = ExtLabelEncoder(classes=classes,
                                   missing_label=missing_label)
        self.missing_label = self._le.missing_label

        # Store cost matrix which will be checked later.
        self.cost_matrix = cost_matrix

        # Store and check random state.
        self.random_state = check_random_state(random_state)

        # Setup classifier if 'classes' is provided as parameter.
        if classes is not None:
            self.classes_ = self._le.classes_
            self.F_components_ = np.zeros(
                (self.mixture_model.n_components, len(self.classes_)))
            if self.cost_matrix is None:
                self.cost_matrix = 1 - np.eye(len(self.classes_))
            self.cost_matrix = check_cost_matrix(self.cost_matrix,
                                                 len(self.classes_))

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
            It contains the weights of the training samples' class labels. It
            must have the same shape as y.

        Returns
        -------
        self: CMM,
            The CMM is fitted on the training data.
        """
        # Check input parameters.
        X = check_array(X)
        self._check_n_features(X, reset=True)
        y = self._le.fit_transform(y)
        check_consistent_length(X, y)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight,
                                        force_all_finite=False,
                                        ensure_2d=False)

        # Refit model if desired.
        if self._refit:
            self.mixture_model.fit(X)

        # Counts number of votes per class label for each sample.
        self.classes_ = self._le.classes_
        V = compute_vote_vectors(y=y, w=sample_weight,
                                 classes=np.arange(len(self.classes_)))

        # Stores responsibility for every given sample of training set.
        R = self.mixture_model.predict_proba(X)

        # Stores class frequency estimates per component.
        self.F_components_ = R.T @ V

        # Update cost matrix.
        self.cost_matrix = 1 - np.eye(len(
            self.classes_)) if self.cost_matrix is None else self.cost_matrix
        self.cost_matrix = check_cost_matrix(self.cost_matrix,
                                             len(self.classes_))

        return self

    def predict_freq(self, X):
        """Return class frequency estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        F : array-like, shape (n_samples, classes)
            The class frequency estimates of the input samples. Classes are
            ordered by lexicographic order.
        """
        check_is_fitted(self, ['F_components_', 'classes_'])
        X = check_array(X)
        if np.sum(self.F_components_) > 0:
            self._check_n_features(X, reset=False)
            D = np.exp(-np.array(
                [cdist(X, [self.mixture_model.means_[j]], metric='mahalanobis',
                       VI=self.mixture_model.precisions_[j]).ravel()
                 for j in range(self.mixture_model.n_components)])).T
            F = D @ self.F_components_
            D /= np.sum(D, axis=1, keepdims=True)
        else:
            F = np.zeros((len(X), len(self.classes_)))
        return F

    def predict_proba(self, X):
        """Return probability estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # normalizing class frequency estimates of each sample
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
        costs = np.dot(P, self.cost_matrix)
        return self._le.inverse_transform(
            rand_argmin(costs, random_state=self.random_state, axis=1))
