import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_random_state, check_array


class PWC(BaseEstimator, ClassifierMixin):
    """PWC
    The Parzen window classifier (PWC) is a simple and probabilistic classifier. This classifier is based on a
    non-parametric density estimation obtained by applying a kernel function.

    Parameters
    ----------
    n_classes: int,
        This parameter indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    kwargs: str,
        Any further parameters are passed directly to the kernel function.

    Attributes
    ----------
    n_classes: int,
        This parameters indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    kwargs: str,
        Any further parameters are passed directly to the kernel function.
    X: array-like, shape (n_samples, n_features)
        The sample matrix X is the feature matrix representing the samples.
    y: array-like, shape (n_samples) or (n_samples, n_outputs)
        It contains the class labels of the training samples.
        The number of class labels may be variable for the samples.
    Z: array-like, shape (n_samples, n_classes)
        The class labels are represented by counting vectors. An entry Z[i,j] indicates how many class labels of class j
        were provided for training sample i.

    References
    ----------
    O. Chapelle, "Active Learning for Parzen Window Classifier",
    Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    def __init__(self, n_classes, metric='rbf', n_neighbors=None, random_state=None, **kwargs):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 0:
            raise ValueError("The parameter 'n_classes' must be a positive integer.")
        self.metric_ = str(metric)
        if self.metric_ not in KERNEL_PARAMS.keys():
            raise ValueError("The parameter 'metric' must be a in {}".format(KERNEL_PARAMS.keys()))
        self.n_neighbors_ = int(n_neighbors) if n_neighbors is not None else n_neighbors
        if self.n_neighbors_ is not None and self.n_neighbors_ <= 0:
            raise ValueError("The parameter 'n_neighbors' must be a positive integer.")
        self.random_state_ = check_random_state(random_state)
        self.kwargs_ = kwargs
        self.X_ = None
        self.y_ = None
        self.Z_ = None

        # include cost matrix
        self.C_ = kwargs.pop('C', 1 - np.eye(self.n_classes_))

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where missing labels are
            represented by np.nan.

        Returns
        -------
        self: PWC,
            The PWC is fitted on the training data.
        """
        if np.size(X) > 0:
            self.X_ = check_array(X)
            self.y_ = check_array(y, ensure_2d=False, force_all_finite=False).astype(int)

            # convert labels to count vectors
            self.Z_ = np.zeros((np.size(X, 0), self.n_classes_))
            for i in range(np.size(self.Z_, 0)):
                self.Z_[i, self.y_[i]] += 1

        return self

    def predict_freq(self, X, **kwargs):
        return self.predict_proba(X, normalize=False, **kwargs)

    def predict_freq_seqal(self, X, y, X_cand, classes, X_eval):
        # freq_cand          (n_cand, n_classes)
        # pred_eval          (n_eval)
        # freq_eval_new_mat  (n_cand, n_classes, n_eval, n_classes),
        # pred_eval_new_mat  (n_cand, n_classes, n_eval)

        self.fit(X, y)
        freq_cand = self.predict_freq(X_cand)

        K = pairwise_kernels(np.vstack([X, X_cand]), X_eval, metric=self.metric_, **self.kwargs_)
        K_X = K[:len(X), :]
        K_X_cand = K[len(X):, :]

        # freq_eval          (n_eval, n_classes)
        if len(y) == 0:
            freq_eval = np.zeros([len(K_X), len(classes)])
        else:
            one_hot_y = np.eye(len(classes))[y]
            freq_eval = K_X.T @ one_hot_y

        # pred_eval          (n_eval)
        # freq_eval_new_mat  (n_cand, n_classes, n_eval, n_classes),
        freq_eval_new_mat = np.tile(np.tile(freq_eval, [len(classes), 1, 1]), [len(X_cand), 1, 1, 1])

        for y_ in classes:
            freq_eval_new_mat[:, y_, :, y_] += K_X_cand

        return freq_cand, freq_eval, freq_eval_new_mat

    def predict_proba(self, X, **kwargs):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.
        C: array-like, shape (n_classes, n_classes)
            Classification cost matrix.

        Returns
        -------
        P:  array-like, shape (t_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # if normalize is false, the probabilities are frequency estimates
        normalize = kwargs.pop('normalize', True)

        # no training data -> random prediction
        if self.X_ is None or np.size(self.X_, 0) == 0:
            if normalize:
                return np.full((np.size(X, 0), self.n_classes_), 1. / self.n_classes_)
            else:
                return np.zeros((np.size(X, 0), self.n_classes_))

        # calculating metric matrix
        if self.metric_ == 'precomputed':
            K = X
        else:
            K = pairwise_kernels(X, self.X_, metric=self.metric_, **self.kwargs_)

        if self.n_neighbors_ is None:
            # calculating labeling frequency estimates
            P = K @ self.Z_
        else:
            if np.size(self.X_, 0) < self.n_neighbors_:
                n_neighbors = np.size(self.X_, 0)
            else:
                n_neighbors = self.n_neighbors_
            indices = np.argpartition(K, -n_neighbors, axis=1)[:, -n_neighbors:]
            P = np.empty((np.size(X, 0), self.n_classes_))
            for i in range(np.size(X, 0)):
                P[i, :] = K[i, indices[i]] @ self.Z_[indices[i], :]

        if normalize:
            # normalizing probabilities of each sample
            normalizer = np.sum(P, axis=1)
            P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
            P[normalizer == 0, :] = [1 / self.n_classes_] * self.n_classes_
            # normalizer[normalizer == 0.0] = 1.0
            # for y_idx in range(self.n_classes):
            #     P[:, y_idx] /= normalizer

        return P

    def predict(self, X, **kwargs):
        """
        Return class label predictions for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y:  array-like, shape = [n_samples]
            Predicted class labels class.
        """
        C = kwargs.pop('C', None)

        if C is None:
            C = np.ones((self.n_classes_, self.n_classes_))
            np.fill_diagonal(C, 0)

        P = self.predict_proba(X, normalize=True)
        return self._rand_arg_min(np.dot(P, C), axis=1)

    def reset(self):
        """
        Reset fitted parameters.
        """
        self.X_ = None
        self.y_ = None
        self.Z_ = None
        self.random_state_ = self.random_state_

    def _rand_arg_min(self, arr, axis=1):
        """
        Returns index of minimal element per given axis. In case of a tie, the index is chosen randomly.

        Parameters
        ----------
        arr: array-like
        Array whose minimal elements' indices are determined.
        axis: int
        Indices of minimal elements are determined along this axis.

        Returns
        -------
        min_indices: array-like
        Indices of minimal elements.
        """
        arr_min = arr.min(axis, keepdims=True)
        tmp = self.random_state_.uniform(low=1, high=2, size=arr.shape) * (arr == arr_min)
        return tmp.argmax(axis)
