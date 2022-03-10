"""
Logistic Regression for Multiple Annotators
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>
#         Timo Sturm <timo.sturm@student.uni-kassel.de>

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal as multi_normal
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from ...base import SkactivemlClassifier, AnnotatorModelMixin
from ...utils import (
    MISSING_LABEL,
    compute_vote_vectors,
    rand_argmax,
    ext_confusion_matrix,
)


class AnnotatorLogisticRegression(SkactivemlClassifier, AnnotatorModelMixin):
    """AnnotatorLogisticRegression

    Logistic Regression based on Raykar [1] is a classification algorithm that
    learns from multiple annotators. Besides, building a model for the
    classification task, the algorithm estimates the performance of the
    annotators. The performance of an annotator is assumed to only depend on
    the true label of a sample and not on the sample itself. Each annotator is
    assigned a confusion matrix, where each row is normalized. This contains
    the bias of the annotators decisions. These estimated biases are then
    used to refine the classifier itself.

    The classifier also supports a bayesian view on the problem, for this a
    prior distribution over an annotator's confusion matrix is assumed. It also
    assumes a prior distribution over the classifiers weight vectors
    corresponding to a regularization.

    Parameters
    ----------
    tol : float, default=1.e-2
        Threshold for stopping the EM-Algorithm, if the change of the
        expectation value between two steps is smaller than tol, the fit
        algorithm stops.
    max_iter : int, default=100
        The maximum number of iterations of the EM-algorithm to be performed.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to input samples.
    annot_prior_full : int or float or array-like, default=1
        This parameter determines A as the Dirichlet prior for each annotator l
        (i.e., A[l] = annot_prior_full * np.ones(n_classes, n_classes) for
        numeric or A[l] = annot_prior_full[l] * np.ones(n_classes, n_classes)
        for array-like parameter). A[l,i,j] is the estimated number of times.
        annotator l has provided label j for an instance of true label i.
    annot_prior_diag : int or float or array-like, default=0
        This parameter adds a value to the diagonal of A[l] being the Dirichlet
        prior for annotator l (i.e., A[l] += annot_prior_diag *
        np.eye(n_classes) for numeric or A[l] += annot_prior_diag[l] *
        np.ones(n_classes) for array-like parameter). A[l,i,j] is the estimated
        number of times annotator l has provided label j for an instance of
        true label i.
    weights_prior : int or float, default=1
        Determines Gamma as the inverse covariance matrix of the
        prior distribution for every weight vector
        (i.e., Gamma=weights_prior * np.eye(n_features)).
        As default, the identity matrix is used for each weight vector.
    solver : str or callable, default='Newton-CG'
        Type of solver.  Should be 'Nelder-Mead', 'Powell', 'CG',
        'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
        'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov',
        or custom - a callable object. See scipy.optimize.minimize for more
        information.
    solver_dict : dictionary, default=None
        Additional solver options passed to scipy.optimize.minimize. If None,
        {'maxiter': 5} is passed.
    classes : array-like of shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with cost_matrix[i,j] indicating cost of predicting class
        classes[j]  for a sample of class classes[i]. Can be only set, if
        classes is not none.
    random_state : int or RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.


    Attributes
    ----------
    n_annotators_ : int
        Number of annotators.
    W_ : numpy.ndarray of shape (n_features, n_classes)
        The weight vectors of the logistic regression model.
    Alpha_ : numpy.ndarray of shape (n_annotators, n_classes, n_classes)
        This is a confusion matrix for each annotator, where each
        row is normalized. `Alpha_[l,k,c]` describes the probability
        that annotator l provides the class label c for a sample belonging
        to class k.
    classes_ : array-like of shape (n_classes)
        Holds the label for each class after fitting.
    cost_matrix_ : array-like of shape (classes, classes)
        Cost matrix with C[i,j] indicating cost of predicting class classes_[j]
        for a sample of class classes_[i].

    References
    ----------
    .. [1] `Raykar, V. C., Yu, S., Zhao, L. H., Valadez, G. H., Florin, C.,
       Bogoni, L., & Moy, L. (2010). Learning from crowds. Journal of Machine
       Learning Research, 11(4).`_
    """

    def __init__(
            self,
            tol=1.0e-2,
            max_iter=100,
            fit_intercept=True,
            annot_prior_full=1,
            annot_prior_diag=0,
            weights_prior=1,
            solver="Newton-CG",
            solver_dict=None,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.annot_prior_full = annot_prior_full
        self.annot_prior_diag = annot_prior_diag
        self.weights_prior = weights_prior
        self.solver = solver
        self.solver_dict = solver_dict

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
        self: AnnotatorLogisticRegression,
            The AnnotatorLogisticRegression is fitted on the training data.
        """
        # Check input data.
        X, y, sample_weight = self._validate_data(
            X=X, y=y, sample_weight=sample_weight, y_ensure_1d=False
        )
        self._check_n_features(X, reset=True)

        # Ensure value of 'tol' to be positive.
        if not isinstance(self.tol, float):
            raise TypeError(
                "`tol` must be an instance of float, not {}.".format(
                    type(self.tol)
                )
            )
        if self.tol <= 0:
            raise ValueError("`tol`= {}, must be > 0.".format(self.tol))

        # Ensure value of 'max_iter' to be positive.
        if not isinstance(self.max_iter, int):
            raise TypeError(
                "`max_iter` must be an instance of int, not {}.".format(
                    type(self.max_iter)
                )
            )
        if self.max_iter <= 0:
            raise ValueError(
                "`max_iter`= {}, must be an integer >= 1.".format(self.tol)
            )

        if not isinstance(self.fit_intercept, bool):
            raise TypeError(
                "'fit_intercept' must be of type 'bool', got {}".format(
                    type(self.fit_intercept)
                )
            )

        solver_dict = (
            {"maxiter": 5} if self.solver_dict is None else self.solver_dict
        )

        # Check weights prior.
        if not isinstance(self.weights_prior, (int, float)):
            raise TypeError(
                "'weights_prior' must be of a positive 'int' or "
                "'float', got {}".format(type(self.weights_prior))
            )
        if self.weights_prior < 0:
            raise ValueError(
                "'weights_prior' must be of a positive 'int' or "
                "'float', got {}".format(self.weights_prior)
            )

        # Check for empty training data.
        if self.n_features_in_ is None:
            return self

        if len(y.shape) != 2:
            raise ValueError(
                "`y` must be an array-like of shape "
                "`(n_samples, n_annotators)`."
            )

        # Insert bias, if 'fit_intercept' is set to 'True'.
        if self.fit_intercept:
            X = np.insert(X, 0, values=1, axis=1)

        # Ensure sample weights to form a 2d array.
        if sample_weight is None:
            sample_weight = np.ones_like(y)

        # Set auxiliary variables.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.n_annotators_ = y.shape[1]
        I = np.eye(n_classes)

        # Convert Gamma to matrix, if it is a number:
        Gamma = self.weights_prior * np.eye(n_features)
        all_zeroes = not np.any(Gamma)
        Gamma_tmp = Gamma if all_zeroes else np.linalg.inv(Gamma)

        # Check input 'annot_prior_full' and 'annot_prior_diag'.
        annot_prior = []
        for name, prior in [
            ("annot_prior_full", self.annot_prior_full),
            ("annot_prior_diag", self.annot_prior_diag),
        ]:
            if isinstance(prior, int or float):
                prior_array = np.ones(self.n_annotators_) * prior
            else:
                prior_array = column_or_1d(prior)
            if name == "annot_prior_full":
                is_invalid_prior = np.sum(prior_array <= 0)
            else:
                is_invalid_prior = np.sum(prior_array < 0)
            if len(prior_array) != self.n_annotators_ or is_invalid_prior:
                raise ValueError(
                    "'{}' must be either 'int', 'float' or "
                    "array-like with positive values and shape "
                    "(n_annotators), got {}".format(name, prior)
                )
            annot_prior.append(prior_array)

        # Set up prior matrix for each annotator.
        A = np.ones((self.n_annotators_, n_classes, n_classes))
        for a in range(self.n_annotators_):
            A[a] *= annot_prior[0][a]
            A[a] += np.eye(n_classes) * annot_prior[1][a]

        # Init Mu (i.e., estimates of true labels) with (weighted) majority
        # voting.
        Mu = compute_vote_vectors(
            y=y,
            classes=np.arange(n_classes),
            missing_label=-1,
            w=sample_weight,
        )
        Mu_sum = np.sum(Mu, axis=1)
        is_zero = Mu_sum == 0
        Mu[~is_zero] /= Mu_sum[~is_zero, np.newaxis]
        Mu[is_zero] = 1 / n_classes

        # Set initial weights.
        self.W_ = np.zeros((n_features, n_classes))

        # Use majority vote to initialize alpha, alpha_j is the confusion
        # matrix of annotator j.
        y_majority = rand_argmax(Mu, random_state=self.random_state, axis=1)
        self.Alpha_ = ext_confusion_matrix(
            y_true=y_majority,
            y_pred=y,
            normalize="true",
            missing_label=-1,
            classes=np.arange(n_classes),
        )

        # Initialize first expectation to infinity such that
        # |current - new| < tol is False.
        current_expectation = -np.inf

        # Execute expectation maximization (EM) algorithm.
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            # E-step:
            P = softmax(X @ self.W_, axis=1)
            V = self._calc_V(y, self.Alpha_)
            Mu = self._calc_Mu(V, P)
            new_expectation = self._calc_expectation(
                Mu, P, V, Gamma, A, self.Alpha_, self.W_
            )

            # Stop EM, if it converges (to a local maximum).
            if (
                    current_expectation == new_expectation
                    or (new_expectation - current_expectation) < self.tol
            ):
                break

            # Update expectation value.
            current_expectation = new_expectation

            # M-Step:
            self._Alpha = self._calc_Alpha(y, Mu, A, sample_weight)

            def error(w):
                """
                Evaluate cross-entropy error of weights for scipy.minimize.

                Parameters
                ----------
                w : ndarray, shape (n_features * n_classes)
                    Weights for which cross-entropy error is to be computed.

                Returns
                -------
                G : flaot
                    Computed cross-entropy error.
                """
                W = w.reshape(n_features, n_classes)
                P_W = softmax(X @ W, axis=1)
                prior_W = 0
                for c_idx in range(n_classes):
                    prior_W += multi_normal.logpdf(
                        x=W[:, c_idx], cov=Gamma_tmp, allow_singular=True
                    )
                log = np.sum(Mu * np.log(P_W * V + np.finfo(float).eps))
                log += prior_W
                return -log / n_samples

            def grad(w):
                """
                Compute gradient of error function for scipy.minimize.

                Parameters
                ----------
                w : ndarray, shape (n_features * n_classes)
                    Weights whose gradient is to be computed.

                Returns
                -------
                G : narray, shape (n_features * n_classes)
                    Computed gradient of weights.
                """
                W = w.reshape(n_features, n_classes)
                P_W = softmax(X @ W, axis=1)
                G = (X.T @ (P_W - Mu) + Gamma @ W).ravel()
                return G / n_samples

            def hessian(w):
                """
                Compute Hessian matrix of error function for scipy.minimize.

                Parameters
                ----------
                w : numpy.ndarray, shape (n_features * n_classes)
                    Weights whose Hessian matrix is to be computed.

                Returns
                -------
                H : numpy.narray, shape (n_features * n_classes,
                n_features * n_classes)
                    Computed Hessian matrix of weights.
                """
                W = w.reshape(n_features, n_classes)
                H = np.empty((n_classes * n_features, n_classes * n_features))
                P_W = softmax(X @ W, axis=1)
                for k in range(n_classes):
                    for j in range(n_classes):
                        diagonal = P_W[:, j] * (I[k, j] - P_W[:, k])
                        D = np.diag(diagonal)
                        H_kj = X.T @ D @ X + Gamma
                        H[
                        k * n_features: (k + 1) * n_features,
                        j * n_features: (j + 1) * n_features,
                        ] = H_kj
                return H / n_samples

            with warnings.catch_warnings():
                warning_msg = ".*Method .* does not use Hessian information.*"
                warnings.filterwarnings("ignore", message=warning_msg)
                warning_msg = ".*Method .* does not use gradient information.*"
                warnings.filterwarnings("ignore", message=warning_msg)
                res = minimize(
                    error,
                    x0=self.W_.ravel(),
                    method=self.solver,
                    tol=self.tol,
                    jac=grad,
                    hess=hessian,
                    options=solver_dict,
                )
                self.W_ = res.x.reshape((n_features, n_classes))

            self.n_iter_ += 1

        return self

    def predict_proba(self, X):
        """Return probability estimates for the test data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P : numpy.ndarray of shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to `classes_`.
        """
        # Check test samples.
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        # Prediction without training data.
        if self.n_features_in_ is None:
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)

        # Check whether a bias feature is missing.
        if self.fit_intercept:
            X = np.insert(X, 0, values=1, axis=1)

        # Compute and normalize probabilities.
        P = softmax(X @ self.W_, axis=1)
        return P

    def predict_annotator_perf(self, X):
        """Calculates the probability that an annotator provides the true label
        for a given sample. The true label is hereby provided by the
        classification model. The label provided by an annotator l is based
        on his/her confusion matrix (i.e., attribute `Alpha_[l]`).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P_annot : numpy.ndarray of shape (n_samples, classes)
            `P_annot[i,l]` is the probability, that annotator l provides the
            correct class label for sample `X[i]`.
        """
        # Prediction without training data.
        if self.n_features_in_ is None:
            return np.ones((len(X), 1)) / len(self.classes_)

        # Compute class probabilities.
        P = self.predict_proba(X)

        # Get correctness probabilities for each annotator per class.
        diag_Alpha = np.array(
            [np.diagonal(self._Alpha[j]) for j in range(self._Alpha.shape[0])]
        )

        # Compute correctness probabilities for each annotator per sample.
        P_annot = P @ diag_Alpha.T
        return P_annot

    @staticmethod
    def _calc_V(y, Alpha):
        """Calculates a value used for updating Mu and the expectation.

        Parameters
        ----------
        y: numpy.ndarray of shape (n_samples, n_annotators)
            The class labels provided by the annotators for all samples.
        Alpha: numpy.ndarray of shape (n_annotators, n_classes, n_classes)
            annot_prior vector (n_annotators, n_classes, n_classes) containing
            the new estimates for Alpha. This is effectively a confusion matrix
            for each annotator, where each row is normalized.

        Returns
        -------
        out: numpy.ndarray
            Vector of shape (n_samples, n_classes).
        """
        n_samples, _, n_classes = (
            y.shape[0],
            y.shape[1],
            Alpha.shape[1],
        )
        V = np.ones((n_samples, n_classes))

        for c in range(n_classes):
            for k in range(n_classes):
                y_is_k = y == k
                V[:, c] *= np.prod(Alpha[:, c, k] ** y_is_k, axis=1)

        return V

    @staticmethod
    def _calc_Alpha(y, Mu, A, sample_weight):
        """Calculates the class-dependent performance estimates of the
        annotators.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples, n_annotators)
            The class labels provided by the annotators for all samples.
        Mu : numpy.ndarray of shape (n_samples, n_classes)
            Mu[i,k] contains the probability of a sample X[i] to be of class
            classes_[k] estimated according to the EM-algorithm.
        A : numpy.ndarray of shape (n_annotators, n_classes, n_classes)
            A[l,i,j] is the estimated number of times.
            annotator l has provided label j for an instance of true label i.
        sample_weight : numpy.ndarray of shape (n_samples, n_annotators)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.

        Returns
        ----------
        new_Alpha : numpy.ndarray of shape
        (n_annotators, n_classes, n_classes)
            This is a confusion matrix for each annotator, where each
            row is normalized. `new_Alpha[l,k,c]` describes the probability
            that annotator l provides the class label c for a sample belonging
            to class k.
        """
        n_annotators, n_classes = y.shape[1], Mu.shape[1]
        new_Alpha = np.zeros((n_annotators, n_classes, n_classes))

        not_nan_y = ~np.isnan(y)
        for j in range(n_annotators):
            # Only take those rows from Y, where Y is not NaN:
            y_j = np.eye(n_classes)[y[not_nan_y[:, j], j].astype(int)]
            w_j = sample_weight[not_nan_y[:, j], j].reshape(-1, 1)
            new_Alpha[j] = (Mu[not_nan_y[:, j]].T @ (w_j * y_j)) + A[j] - 1

        # Lazy normalization: (The real normalization factor
        # (sum_i=1^N mu_i,c + sum_k=0^K-1 A_j,c,k - K) is omitted here)
        with np.errstate(all="ignore"):
            new_Alpha = new_Alpha / new_Alpha.sum(axis=2, keepdims=True)
            new_Alpha = np.nan_to_num(new_Alpha, nan=1.0 / n_classes)

        return new_Alpha

    @staticmethod
    def _calc_Mu(V, P):
        """Calculates the new estimate for Mu, using Bayes' theorem.

        Parameters
        ----------
        V : numpy.ndarray, shape (n_samples, n_classes)
            Describes an intermediate result.
        P : numpy.ndarray, shape (n_samples, n_classes)
            P[i,k] contains the probabilities of sample X[i] belonging to class
            classes_[k], as estimated by the classifier
            (i.e., sigmoid(W.T, X[i])).

        Returns
        -------
        new_Mu : numpy.ndarray
            new_Mu[i,k] contains the probability of a sample X[i] to be of
            class classes_[k] estimated according to the EM-algorithm.
        """
        new_Mu = P * V
        new_Mu_sum = np.sum(new_Mu, axis=1)
        is_zero = new_Mu_sum == 0

        new_Mu[~is_zero] /= new_Mu_sum[~is_zero, np.newaxis]
        new_Mu[is_zero] = 1 / P.shape[1]
        return new_Mu

    @staticmethod
    def _calc_expectation(Mu, P, V, Gamma, A, Alpha, W):
        """Calculates the conditional expectation in the E-step of the
        EM-Algorithm, given the observations and the current estimates of the
        classifier.

        Parameters
        ----------
        Mu : numpy.ndarray, shape (n_samples, n_classes)
            Mu[i,k] contains the probability of a sample X[i] to be of class
            classes_[k] estimated according to the EM-algorithm.
        V : numpy.ndarray, shape (n_samples, n_classes)
            Describes an intermediate result.
        P : numpy.ndarray, shape (n_samples, n_classes)
            P[i,k] contains the probabilities of sample X[i] belonging to class
            classes_[k], as estimated by the classifier
            (i.e., sigmoid(W.T, X[i])).

        Returns
        -------
        expectation : float
            The conditional expectation.
        """
        # Evaluate prior of weight vectors.
        all_zeroes = not np.any(Gamma)
        Gamma = Gamma if all_zeroes else np.linalg.inv(Gamma)
        prior_W = np.sum(
            [
                multi_normal.logpdf(x=W[:, k], cov=Gamma, allow_singular=True)
                for k in range(W.shape[1])
            ]
        )

        # Evaluate prior of alpha matrices.
        prior_Alpha = np.sum(
            [
                [
                    dirichlet.logpdf(x=Alpha[j, k, :], alpha=A[j, k, :])
                    for k in range(Alpha.shape[1])
                ]
                for j in range(Alpha.shape[0])
            ]
        )

        # Evaluate log-likelihood for data.
        log_likelihood = np.sum(Mu * np.log(P * V + np.finfo(float).eps))
        expectation = log_likelihood + prior_W + prior_Alpha
        return expectation
