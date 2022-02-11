import numpy as np
from sklearn.metrics import pairwise_kernels

from skactiveml.base import SkactivemlContinuousEstimator
from skactiveml.utils._functions import is_scalar
from skactiveml.utils._label import is_all_labeled


class NormalInverseWishartKernelEstimator(SkactivemlContinuousEstimator):

    def __init__(self, metric='rbf', metric_dict=None, kappa_0=0.1,
                 nu_0='d', mu_0=0.0, S_0=1.0, random_state=None):
        super().__init__(random_state=random_state)
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.S_0 = S_0
        self.metric = metric
        self.metric_dict = {} if metric_dict is None else metric_dict

    def fit(self, X, y, sample_weight=None):
        d_t = y.shape[1]
        d_f = X.shape[1]
        if is_scalar(self.mu_0):
            mu_0 = np.full((1, d_t), self.mu_0)
        else:
            mu_0 = np.array(self.mu_0)
            mu_0 = mu_0.reshape(1, -1)
        X_0 = np.zeros(d_f).reshape(1, -1)
        lbld = is_all_labeled(y)
        self.X_ = np.append(X_0, X[lbld], axis=0)
        self.Y_ = np.append(mu_0, y[lbld], axis=0)
        return self

    def estimate_post(self, X):

        n_cand = X.shape[0]
        n_samples = self.Y_.shape[0]
        d_t = self.Y_.shape[1]

        if self.nu_0 == 'd':
            self.nu_0 = self.Y_.shape[1] + 1

        if is_scalar(self.S_0):
            S_0 = self.S_0 * np.identity(d_t).reshape(1, d_t, d_t)
        else:
            S_0 = np.array(self.S_0).reshape(self.S_0)
            S_0 = S_0.reshape((1, ) + S_0.shape)
        K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_dict)
        K[:, 0] = self.kappa_0

        kappa_L = self.kappa_0 + np.sum(K, axis=1)
        nu_L = self.nu_0 + np.sum(K, axis=1)
        mu_L = 1/kappa_L.reshape(-1, 1) * K @ self.Y_
        diff = self.Y_.reshape(1, n_samples, d_t) - mu_L.reshape(n_cand, 1, d_t)
        scatters = diff.reshape(n_cand, n_samples, d_t, 1) \
            * diff.reshape(n_cand, n_samples, 1, d_t)
        S_L = S_0.reshape(1, d_t, d_t) \
            + np.sum(K.reshape(n_cand, n_samples, 1, 1) * scatters, axis=1)

        return kappa_L, nu_L, mu_L, S_L

    def predict(self, X):
        mu_L, Cov_L = self.estimate_mu_cov(X)
        return mu_L

    def estimate_random_variates(self, X, n_rvs):
        pass

    def estimate_mu_cov(self, X):
        kappa_L, nu_L, mu_L, S_L = self.estimate_post(X)
        n_cand = X.shape[0]
        d_tar = self.Y_.shape[1]
        return mu_L, ((kappa_L+1)/((nu_L - d_tar - 1)*kappa_L))\
            .reshape(n_cand, 1, 1)*S_L
