"""
Classifier based on a Gaussian Mixture Model
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>

from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils.validation import check_array, \
    check_is_fitted, NotFittedError

from ..base import ClassFrequencyEstimator
from ..utils import MISSING_LABEL, compute_vote_vectors


class CMM(ClassFrequencyEstimator):
    """CMM

    Classifier mixture model (CMM) is a generative classifier based on a
    (Bayesian) Gaussian mixture model (GMM).

    Parameters
    ----------
    mixture_model : {GaussianMixture, BayesianGaussianMixture, None},
    default=None
        (Bayesian) Gaussian Mixture model that is trained with unsupervised
        algorithm on train data. If the initial mixture model is not fitted, it
        will be refitted in each call of the 'fit' method. If None,
        mixture_model=BayesianMixtureModel(n_components=n_classes) will be
        used.
    weight_mode : {'responsibilities', 'similarities'},
        default='responsibilities'
        Determines whether the responsibilities outputted by the
        `mixture_model` or the exponentials of the Mahalanobis distances as
        similarities are used to compute the class frequency estimates.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : {scalar, string, np.nan, None}, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        classes is not none.
    class_prior : float | array-like, shape (n_classes), optional (default=0)
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class.
    random_state : int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes)
        Holds the label for each class after fitting.
    class_prior : np.ndarray, shape (n_classes)
        Prior observations of the class frequency estimates. The entry
        `class_prior_[i]` indicates the non-negative prior number of samples
        belonging to class `classes_[i]`.
    cost_matrix_ : np.ndarray, shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    F_components_ : numpy.ndarray, shape (n_components, n_classes)
        `F[j,c]` is a proxy for the number of sample of class c belonging to
        component j.
    mixture_model_ : {GaussianMixture, BayesianGaussianMixture}
        (Bayesian) Gaussian Mixture model that is trained with unsupervised
        algorithm on train data.
    """
    def __init__(self, mixture_model=None, weight_mode='responsibilities',
                 classes=None, missing_label=MISSING_LABEL, cost_matrix=None,
                 class_prior=0.0, random_state=None):
        super().__init__(classes=classes, missing_label=missing_label,
                         cost_matrix=cost_matrix, random_state=random_state)
        self.class_prior = class_prior
        self.mixture_model = mixture_model
        self.weight_mode = weight_mode

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
        X, y, sample_weight = self._validate_data(X, y, sample_weight)
        self._check_n_features(X, reset=True)

        # Check mixture model.
        if self.mixture_model is None:
            bgm = BayesianGaussianMixture(n_components=len(self.classes_),
                                          random_state=self._random_state)
            self.mixture_model_ = bgm
        else:
            if not isinstance(self.mixture_model,
                              (GaussianMixture, BayesianGaussianMixture)):
                raise TypeError(
                    "'mixture_model' is of the type '{}' but must be of the "
                    "type 'sklearn.mixture.GaussianMixture' or "
                    "'sklearn.mixture.BayesianGaussianMixture'.".format(
                        type(self.mixture_model)))
            self.mixture_model_ = deepcopy(self.mixture_model)

        # Check weight mode.
        if self.weight_mode not in ['responsibilities', 'similarities']:
            raise ValueError("'weight_mode' must be either `responsibilities` "
                             "or `similarities`, got {} instead."
                             .format(self.weight_mode))

        if self.n_features_in_ is None:
            self.F_components_ = 0
        else:
            # Refit model if desired.
            try:
                check_is_fitted(self.mixture_model_)
            except NotFittedError:
                self.mixture_model_ = self.mixture_model_.fit(X)

            # Counts number of votes per class label for each sample.
            V = compute_vote_vectors(y=y, w=sample_weight,
                                     classes=np.arange(len(self.classes_)))

            # Stores responsibility for every given sample of training set.
            R = self.mixture_model_.predict_proba(X)

            # Stores class frequency estimates per component.
            self.F_components_ = R.T @ V

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
            ordered according to classes_.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        if np.sum(self.F_components_) > 0:
            if self.weight_mode == 'similarities':
                S = np.exp(-np.array(
                    [cdist(X, [self.mixture_model_.means_[j]],
                           metric='mahalanobis',
                           VI=self.mixture_model_.precisions_[j]).ravel()
                     for j in range(self.mixture_model_.n_components)])).T
            else:
                S = self.mixture_model_.predict_proba(X)
            F = S @ self.F_components_
        else:
            F = np.zeros((len(X), len(self.classes_)))
        return F