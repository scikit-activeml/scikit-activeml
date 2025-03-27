"""
Logistic Regression for Multiple Annotators
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>
#         Timo Sturm <timo.sturm@student.uni-kassel.de>

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    column_or_1d,
)

from ...base import SkactivemlClassifier, AnnotatorModelMixin
from ...utils import (
    MISSING_LABEL,
    compute_vote_vectors,
    is_labeled,
    check_scalar,
    check_n_features,
)


class AnnotatorLogisticRegression(SkactivemlClassifier, AnnotatorModelMixin):
    """Logistic Regression for Crowds

    Logistic Regression based on Raykar [1]_ is a classification algorithm
    that learns from multiple annotators. Besides, building a model for the
    classification task, the algorithm estimates the performance of the
    annotators. The performance of an annotator is assumed to only depend on
    the true label of a sample and not on the sample itself. Each annotator is
    assigned a confusion matrix, where each row is normalized. This contains
    the bias of the annotators decisions. These estimated biases are then
    used to refine the classifier itself.

    The classifier also supports a bayesian view on the problem, for this a
    prior distribution over an annotator's confusion matrix is assumed. It also
    assumes a prior distribution over the classifiers' weight vectors
    corresponding to a regularization.

    Parameters
    ----------
    tol : float, default=0.0001
        Threshold for stopping the EM-Algorithm and the optimization of the
        the logistic regression weights. If the change of the respective value
        between two steps is smaller than `tol`, the respective algorithm
        stops.
    max_iter : int, default=100
        The maximum number of iterations of the EM-algorithm to be performed.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to input samples.
    annot_prior_full : int or float or array-like, default=1
        Determines `A` as the Dirichlet prior for each annotator `l`
        (i.e., `A[l] = annot_prior_full * np.ones(n_classes, n_classes)` for
        numeric or `A[l] = annot_prior_full[l] * np.ones(n_classes, n_classes)`
        for array-like parameter). `A[l,i,j]` is the estimated number of times.
        annotator `l` has provided label `j` for a sample of true label `i`.
    annot_prior_diag : int or float or array-like, default=0
        Adds a value to the diagonal of `A[l]` being the Dirichlet
        prior for annotator `l` (i.e., `A[l] += annot_prior_diag *
        np.eye(n_classes)` for numeric or `A[l] += annot_prior_diag[l] *
        np.ones(n_classes)` for array-like parameter). `A[l,i,j]` is the
        estimated number of times annotator `l` has provided label `j` for
        a sample of true label `i`.
    weights_prior : int or float, default=1
        Determines Gamma as the inverse covariance matrix of the
        prior distribution for every weight vector
        (i.e., `Gamma=weights_prior * np.eye(n_features)`).
        As default, the identity matrix is used for each weight vector.
    solver : str or callable, default='L-BFGS-B'
        Type of solver.  Should be 'Nelder-Mead', 'Powell', 'CG',
        'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
        'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov',
        or custom - a callable object. See `scipy.optimize.minimize` for more
        information.
    solver_dict : dictionary, default=None
        Additional solver options passed to scipy.optimize.minimize. If `None`,
        {'maxiter': 100} is passed.
    classes : array-like of shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        `classes is not None`.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
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
        that annotator `l` provides the class label `c` for a sample belonging
        to class `k`.
    classes_ : array-like of shape (n_classes,)
        Holds the label for each class after fitting.
    cost_matrix_ : array-like of shape (classes, classes)
        Cost matrix with `C[i,j]` indicating cost of predicting class
        `self.classes_[j]` for a sample of class `classes_[i]`.

    References
    ----------
    .. [1] V. C. Raykar, S. Yu, L. H. Zhao, G. H. Valadez, C. Florin, L.
       Bogoni, and L. Moy. Learning from Crowds. J. Mach. Learn. Res.,
       11(4):1297–1322, 2010.
    """

    def __init__(
        self,
        n_annotators=None,
        tol=0.0001,
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
        self.n_annotators = n_annotators
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.annot_prior_full = annot_prior_full
        self.annot_prior_diag = annot_prior_diag
        self.weights_prior = weights_prior
        self.solver = solver
        self.solver_dict = solver_dict

    def fit(self, X, y, sample_weight=None):
        """Fit the model using `X` as samples and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute `missing_label`.
        sample_weight : array-like of shape (n_samples,) or\
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            It must have the same shape as `y`. Accordingly, the sample
            weights are only used for the initialization of the majority vote
            and the computation of the confusion matrix. It is not supported
            for the update of logistic regression weights and the expectation
            computation.

        Returns
        -------
        self: AnnotatorLogisticRegression,
            The `AnnotatorLogisticRegression` is fitted on the training data.
        """
        # Check input data.
        X, y, sample_weight = self._validate_data(
            X=X, y=y, sample_weight=sample_weight, y_ensure_1d=False
        )

        # Ensure value of 'tol' to be a positive integer.
        if self.n_annotators is not None:
            check_scalar(
                self.n_annotators,
                "n_annotators",
                min_val=1,
                min_inclusive=True,
                target_type=int,
            )

        # Ensure value of 'tol' to be positive.
        check_scalar(
            self.tol,
            "tol",
            min_val=0,
            min_inclusive=False,
            target_type=(int, float),
        )

        # Ensure value of 'max_iter' to be positive.
        check_scalar(
            self.max_iter,
            "max_iter",
            min_val=1,
            min_inclusive=True,
            target_type=int,
        )

        # Ensure value of 'fit_intercept' to be boolean.
        check_scalar(self.fit_intercept, "fit_intercept", target_type=bool)

        # Ensure value of 'solver_dict' to be a dictionary.
        solver_dict = (
            {"maxiter": 100} if self.solver_dict is None else self.solver_dict
        )
        check_scalar(solver_dict, "solver_dict", target_type=dict)

        # Ensure value of 'weights_prior' to be non-negative.
        check_scalar(
            self.weights_prior,
            "weights_prior",
            min_val=0,
            min_inclusive=True,
            target_type=(int, float),
        )

        # Set auxiliary variables.
        n_classes = len(self.classes_)
        if self.n_features_in_ is not None:
            if len(y.shape) != 2:
                raise ValueError(
                    "`y` must be an array-like of shape "
                    "`(n_samples, n_annotators)`."
                )
            self.n_annotators_ = y.shape[1]
        else:
            if self.n_annotators is None:
                raise ValueError(
                    "`y` cannot be empty, if `n_annotators` is None."
                )
            self.n_annotators_ = self.n_annotators

        # Check consistent number of annotators.
        if (
            self.n_annotators is not None
            and self.n_annotators != self.n_annotators_
        ):
            raise ValueError(
                f"`n_annotators={self.n_annotators}` does not match "
                f"{self.n_annotators_} as the number of columns in `y`."
            )

        # Check input 'annot_prior_full' and 'annot_prior_diag'.
        annot_prior = []
        for name, prior in [
            ("annot_prior_full", self.annot_prior_full),
            ("annot_prior_diag", self.annot_prior_diag),
        ]:
            if isinstance(prior, (int, float)):
                prior_array = np.ones(self.n_annotators_) * prior
            else:
                prior_array = column_or_1d(prior)
            if name == "annot_prior_full":
                is_invalid_prior = np.sum(prior_array <= 0)
            else:
                is_invalid_prior = np.sum(prior_array < 0)
            if len(prior_array) != self.n_annotators_ or is_invalid_prior:
                raise ValueError(
                    f"'{name}' must be either 'int', 'float' or "
                    f"array-like with positive values and shape "
                    f"(n_annotators), got {prior}"
                )
            annot_prior.append(prior_array)

        # Set up prior matrix for each annotator.
        A = np.ones((self.n_annotators_, n_classes, n_classes))
        for a in range(self.n_annotators_):
            A[a] *= annot_prior[0][a]
            A[a] += np.eye(n_classes) * annot_prior[1][a]
        A_obs = A - 1
        A_obs_sum = np.sum(A_obs, axis=-1, keepdims=True)
        A_norm = np.divide(
            A_obs,
            A_obs_sum,
            out=np.full_like(A_obs, 1 / n_classes, dtype=float),
            where=A_obs_sum != 0,
        )

        # Fallback, if empty training data has been provided.
        if self.n_features_in_ is None:
            self.W_ = None
            self.Alpha_ = A_norm
            return self

        # Remove samples without labels.
        is_lbld = is_labeled(y, missing_label=-1).any(axis=-1)
        X = X[is_lbld]
        y = y[is_lbld]
        n_samples = X.shape[0]
        is_lbld = is_labeled(y, missing_label=-1)

        # Insert bias, if 'fit_intercept' is set to 'True'.
        n_weights = self.n_features_in_
        if self.fit_intercept:
            n_weights += 1
            X = np.insert(X, 0, values=1, axis=1)

        # Ensure sample weights to form a 2d array.
        if sample_weight is None:
            sample_weight = np.ones_like(y)

        # Set initial weights.
        self.W_ = np.zeros((n_weights, n_classes))

        if n_samples == 0:
            self.Alpha_ = A_norm
            return self

        # Initialize first expectation to infinity such that
        # |current - new| < tol is False.
        current_expectation = -np.inf
        n_iter = 0

        # Execute expectation maximization (EM) algorithm.
        while n_iter < self.max_iter:
            # -----------------------------E-STEP------------------------------
            # Compute probabilistic predictions.
            P = softmax(X @ self.W_, axis=1)

            # Estimate latent ground truth labels.
            if n_iter == 0:
                # Initialization via majority voting.
                Mu = compute_vote_vectors(
                    y=y,
                    classes=np.arange(n_classes),
                    missing_label=-1,
                    w=sample_weight,
                )
                Mu /= np.sum(Mu, axis=1, keepdims=True)
            else:
                # Use current model parameters to estimate ground truth labels.
                U = np.ones((n_samples, n_classes))
                for c in range(n_classes):
                    for k in range(n_classes):
                        y_is_k = y == k
                        U[:, c] *= np.prod(
                            self.Alpha_[:, c, k] ** y_is_k, axis=1
                        )
                Mu = P * U
                Mu_sum = Mu.sum(axis=1, keepdims=True)
                Mu = np.divide(
                    Mu,
                    Mu_sum,
                    out=np.full_like(Mu, 1 / n_classes, dtype=float),
                    where=Mu_sum != 0,
                )

                # Compute expectation for current parameters and estimates
                # of the ground truth labels.
                prior_w = -0.5 * self.weights_prior * np.sum(self.W_**2)
                prior_alpha = np.sum(
                    (A - 1) * np.log(self.Alpha_ + np.finfo(float).eps)
                )
                log_likelihood = np.sum(
                    Mu * np.log(P * U + np.finfo(float).eps)
                )
                new_expectation = log_likelihood + prior_w + prior_alpha

                # Stop in the case of convergence.
                if np.abs(new_expectation - current_expectation) < self.tol:
                    break

                # Otherwise, update the current expectation value.
                current_expectation = new_expectation

            # -----------------------------M-STEP------------------------------
            # Update the confusion matrices.
            Alpha = np.zeros((self.n_annotators_, n_classes, n_classes))
            for j in range(self.n_annotators_):
                y_j = np.eye(n_classes)[y[is_lbld[:, j], j].astype(int)]
                w_j = sample_weight[is_lbld[:, j], j].reshape(-1, 1)
                Alpha[j] = (Mu[is_lbld[:, j]].T @ (w_j * y_j)) + A[j] - 1
            Alpha_sum = Alpha.sum(axis=-1, keepdims=True)
            self.Alpha_ = np.divide(
                Alpha,
                Alpha_sum,
                out=np.full_like(Alpha, 1 / n_classes, dtype=float),
                where=Alpha_sum != 0,
            )

            def error(w):
                """
                Compute the cross-entropy loss (negative log-posterior) and
                its gradient for a softmax-based classifier with L2
                regularization.

                This function calculates the error as the sum of the negative
                log-likelihood (cross-entropy) of the predicted class
                probabilities relative to the target distribution, and an L2
                regularization penalty on the weights. It also computes the
                gradient of this loss with respect to the weight vector, which
                is returned in flattened form. This formulation is suitable for
                use with optimization routines such as
                `scipy.optimize.minimize`.

                Parameters
                ----------
                w : numpy.ndarray of shape (n_weights * n_classes,)
                    Flattened weight vector. It is reshaped to a
                    (n_weights, n_classes) weight matrix.

                Returns
                -------
                loss : float
                    The computed cross-entropy error (negative log-posterior),
                    including the negative log-likelihood and the
                    L2 regularization penalty.
                grad : numpy.ndarray of shape (n_features * n_classes,)
                    The gradient of the loss with respect to the weight vector,
                    flattened to a one-dimensional array
                """
                # Reshape weights as matrix.
                W = w.reshape((n_weights, n_classes))

                # Compute probabilistic predictions.
                logits = X.dot(W)
                p = softmax(logits, axis=-1)

                # Compute loss for probabilistic predictions.
                loss = -np.sum(Mu * np.log(p + np.finfo(float).eps))

                # Add L2 penalty to loss.
                if self.fit_intercept:
                    loss += 0.5 * self.weights_prior * np.sum(W[1:] ** 2)
                else:
                    loss += 0.5 * self.weights_prior * np.sum(W**2)

                # Compute L2 part of the gradient.
                if self.fit_intercept:
                    reg_grad = np.zeros_like(W)
                    reg_grad[1:, :] = self.weights_prior * W[1:, :]
                else:
                    reg_grad = self.weights_prior * W

                # Compute final gradient.
                grad = X.T.dot(p - Mu) + reg_grad

                return loss, grad.ravel()

            def hessp(w, v):
                """
                Compute the Hessian-vector product for the error function.

                This function computes the product of the Hessian of the error
                function with a given vector, which is useful for second-order
                optimization routines (e.g. in scipy.optimize.minimize when
                a Hessian-vector product is supplied via the 'hessp' argument).
                The error function is defined via a softmax-based prediction
                model with an added regularization term.

                Parameters
                ----------
                w : numpy.ndarray of shape (n_weights * n_classes,)
                    Flattened weight vector. It is reshaped to a
                    (n_weights, n_classes) weight matrix.
                v : numpy.ndarray of shape (n_weights * n_classes,)
                    Flattened vector to be multiplied by the Hessian. It is
                    reshaped to a (n_weights, n_classes) matrix.

                Returns
                -------
                Hv : numpy.ndarray of shape (n_weights * n_classes,)
                    The product of the Hessian (of the error function) with the
                     vector v, returned as a flat array.
                """
                # Reshape weights as matrix.
                W = w.reshape((n_weights, n_classes))

                # Compute probabilistic predictions.
                logits = X.dot(W)
                probas = softmax(logits, axis=-1)

                # Compute intermediate results.
                V = v.reshape((n_weights, n_classes))
                M = X.dot(V)

                # For each sample X[i], compute:
                # R[i,j] = p[i,j] * ( M[i,j] - sum_k p[i,k]*M[i,k] )
                R = probas * (M - np.sum(probas * M, axis=1, keepdims=True))

                if self.fit_intercept:
                    reg_Hv = np.zeros_like(V)
                    reg_Hv[1:, :] = self.weights_prior * V[1:, :]
                else:
                    reg_Hv = self.weights_prior * V

                Hv = X.T.dot(R) + reg_Hv
                return Hv.ravel()

            # Update weights of the logistic regression model.
            with warnings.catch_warnings():
                warning_msg = ".* does not use Hessian.* information.*"
                warnings.filterwarnings("ignore", message=warning_msg)
                warning_msg = ".* does not use gradient information.*"
                warnings.filterwarnings("ignore", message=warning_msg)

                res = minimize(
                    error,
                    x0=np.zeros((n_weights * n_classes)),
                    method=self.solver,
                    tol=self.tol,
                    jac=True,
                    hessp=hessp,
                    options=solver_dict,
                )
                self.W_ = res.x.reshape((n_weights, n_classes))

            # Continue with next iteration.
            n_iter += 1

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
            according to the attribute `self.classes_`.
        """
        # Check test samples.
        check_is_fitted(self)
        X = check_array(X)
        check_n_features(self, X, reset=False)

        # Check whether a bias feature is missing.
        if self.fit_intercept:
            X = np.insert(X, 0, values=1, axis=1)

        # Compute and normalize probabilities.
        if self.W_ is not None:
            P = softmax(X @ self.W_, axis=1)
        else:
            return np.full(
                (len(X), len(self.classes_)), fill_value=1 / len(self.classes_)
            )
        return P

    def predict_annotator_perf(self, X):
        """Calculates the probability that an annotator provides the true label
        for a given sample. The true label is hereby provided by the
        classification model. The label provided by an annotator `l` is based
        on their confusion matrix (i.e., attribute `self.Alpha_[l]`).

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
        # Compute class probabilities.
        P = self.predict_proba(X)

        # Get correctness probabilities for each annotator per class.
        diag_Alpha = np.array(
            [np.diagonal(self.Alpha_[j]) for j in range(self.Alpha_.shape[0])]
        )

        # Compute correctness probabilities for each annotator per sample.
        P_annot = P @ diag_Alpha.T
        return P_annot
