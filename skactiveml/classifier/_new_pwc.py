import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted, check_scalar

from ..utils import rand_argmin, compute_vote_vectors, ExtLabelEncoder


class PWC(BaseEstimator, ClassifierMixin):
    """PWC
    The Parzen window classifier (PWC) is a simple and probabilistic classifier. This classifier is based on a
    non-parametric density estimation obtained by applying a kernel function.

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None (inference during fit)
        List of unique class labels.
    metric : str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors : int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    metric_kwargs : str,
        Any further parameters are passed directly to the kernel function.
    cost_matrix : array-like, shape (classes, classes)
        Cost matrix with C[i,j] indicating cost of predicting class j for a sample of class i.

    Attributes
    ----------
    classes : array-like, shape (n_classes), default=None (inference during fit)
        List of unique class labels.
    metric : str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors : int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    metric_kwargs : str,
        Any further parameters are passed directly to the kernel function.
    cost_matrix : array-like, shape (classes, classes)
        Cost matrix with C[i,j] indicating cost of predicting class j for a sample of class i.
    X_ : array-like, shape (n_samples, n_features)
        The sample matrix X is the feature matrix representing the samples.
    y_ : array-like, shape (n_samples) or (n_samples, n_outputs)
        It contains the class labels of the training samples.
        The number of class labels may be variable for the samples.
    V_ : array-like, shape (n_samples, classes)
        The class labels are represented by counting vectors. An entry V[i,j] indicates how many class labels of class j
        were provided for training sample i.

    References
    ----------
    O. Chapelle, "Active Learning for Parzen Window Classifier",
    Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """
    METRICS = list(KERNEL_PARAMS.keys()) + ['precomputed']

    def __init__(self, classes=None, unlabeled_class=np.nan, metric='rbf', n_neighbors=None, cost_matrix=None,
                 random_state=None, **metric_kwargs):
        self.le = ExtLabelEncoder(classes=classes, unlabeled_class=unlabeled_class)
        self.metric = str(metric)
        if self.metric not in PWC.METRICS:
            raise ValueError("The parameter 'metric' must be a in {}".format(KERNEL_PARAMS.keys()))
        self.n_neighbors = n_neighbors
        if n_neighbors is not None:
            check_scalar(self.n_neighbors, name='n_neighbors', min_val=1, target_type=int)
        self.cost_matrix = check_array(cost_matrix) if cost_matrix is not None else None
        self.random_state = check_random_state(random_state)
        self.metric_kwargs = metric_kwargs
        self.X_ = None
        self.y_ = None
        self.V_ = None

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where missing labels are
            represented the attribute 'unlabeled'.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It must have the same shape as y.

        Returns
        -------
        self: PWC,
            The PWC is fitted on the training data.
        """
        if np.size(X) > 0:
            self.X_ = check_array(X)
            self.y_ = self.le.fit_transform(y)
            if sample_weight is not None:
                sample_weight = check_array(sample_weight, force_all_finite=False, ensure_2d=False)

            # convert labels to count vectors
            self.V_ = compute_vote_vectors(y=self.y_, w=sample_weight, classes=np.arange(len(self.le.classes)))
        else:
            if self.le.classes is None:
                raise ValueError(
                    "You cannot fit a classifier on empty data, if parameter 'classes' has not been specified.")
            self.le.fit(self.le.classes)
            self.V_ = np.array([])

        self.cost_matrix = 1 - np.eye(len(self.le.classes)) if self.cost_matrix is None else self.cost_matrix

        return self

    def predict_freq(self, X):
        """Return class frequency estimates for the input data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        F: array-like, shape (n_samples, classes)
            The class frequency estimates of the input samples. Classes are ordered by lexicographic order.
        """
        check_is_fitted(self.le)
        if np.size(self.V_, 0) == 0:
            return np.zeros((np.size(X, 0), len(self.le.classes)))

        # calculating metric matrix
        if self.metric == 'precomputed':
            K = np.asarray(X)
            if np.size(K, 0) != np.size(X, 0) or np.size(K, 1) != np.size(self.X_, 0):
                raise ValueError("The kernel matrix 'X' must have the shape (n_test_samples, n_train_samples).")
        else:
            K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_kwargs)

        # computing class frequency estimates
        if self.n_neighbors is None or np.size(self.X_, 0) <= self.n_neighbors:
            F = K @ self.V_
        else:
            indices = np.argpartition(K, -self.n_neighbors, axis=1)[:, -self.n_neighbors:]
            F = np.empty((np.size(X, 0), len(self.le.classes)))
            for i in range(np.size(X, 0)):
                F[i, :] = K[i, indices[i]] @ self.V_[indices[i], :]
        return F

    def predict_freq_seqal(self, X, y, X_cand, classes, X_eval):
        # freq_cand          (n_cand, classes)
        # pred_eval          (n_eval)
        # freq_eval_new_mat  (n_cand, classes, n_eval, classes),
        # pred_eval_new_mat  (n_cand, classes, n_eval)
        self.fit(X, y)
        freq_cand = self.predict_freq(X_cand)

        K = pairwise_kernels(np.vstack([X, X_cand]), X_eval, metric=self.metric, **self.metric_kwargs)
        K_X = K[:len(X), :]
        K_X_cand = K[len(X):, :]

        # freq_eval          (n_eval, classes)
        one_hot_y = np.eye(len(classes))[y]
        freq_eval = K_X.T @ one_hot_y

        # pred_eval          (n_eval)
        # freq_eval_new_mat  (n_cand, classes, n_eval, classes),
        freq_eval_new_mat = np.tile(np.tile(freq_eval, [len(classes), 1, 1]), [len(X_cand), 1, 1, 1])

        for y_ in classes:
            freq_eval_new_mat[:, y_, :, y_] += K_X_cand

        return freq_cand, freq_eval, freq_eval_new_mat

    def predict_proba(self, X):
        """Return probability estimates for the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # normalizing probabilities of each sample
        P = self.predict_freq(X)
        normalizer = np.sum(P, axis=1)
        P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
        P[normalizer == 0, :] = [1 / len(self.le.classes)] * len(self.le.classes)
        return P

    def predict(self, X):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        P = self.predict_proba(X)
        costs = np.dot(P, self.cost_matrix)
        return self.le.inverse_transform(rand_argmin(costs, random_state=self.random_state, axis=1))
