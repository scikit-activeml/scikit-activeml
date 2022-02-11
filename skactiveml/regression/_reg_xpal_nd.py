from copy import deepcopy

import numpy as np

from ..base import SingleAnnotPoolBasedQueryStrategy
from ..pool import RandomSampler
from ..regressor.estimator._ngke import NormalGammaKernelEstimator
from ..regressor.estimator._nwke import NormalInverseWishartKernelEstimator
from ..utils import rand_argmax


class RegxPalNd(SingleAnnotPoolBasedQueryStrategy):
    """Probablistic Active Learning Approach for Regression for multiple
    dimensions

    This class implements xPal for Regresssion for mutliple dimensions.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    n_monte_carlo_samples: int, optional (default=1)
        The number of samples that are drawn to expect the xgain from the
        performance boost, if linear is not true.
    post_est: SkactivemlContinuousPosteriorEstimator
        The estimator used to estimate the probability distribution.
    """

    def __init__(self, random_state=None, post_est=None,
                 n_monte_carlo_samples=10):
        super().__init__(random_state=random_state)
        if post_est is None:
            self.posterior_estimator = NormalInverseWishartKernelEstimator()
        else:
            self.posterior_estimator = post_est
        self.n_monte_carlo_samples = n_monte_carlo_samples

    def query(self, X_cand, reg, E, X, y, batch_size=1,
              assume_linear=False, return_utilities=False):

        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        reg: SkactivemlRegressor
            regressor to predict values of X_cand.
        E: array-like, shape (n_evaluator, n_features)
            Evaluation set.
        X: array-like, shape (n_samples, n_features)
            Complete training data set.
        y: array-like, shape (n_samples, n_targets)
            Values of the training data set.
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        assume_linear: bool, optional (default=2)
            Whether the model is assumed to be linear in the y-components of the
            given samples (`y`).
        return_utilities: bool, optional (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (batch_size)
            The index of the queried instance.
        utilities: np.ndarray, shape (batch_size, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        n_candidates = len(X_cand)
        utilities = np.ones(shape=n_candidates)

        if np.all(np.isnan(y)):
            if return_utilities:
                return rand_argmax(utilities), utilities
            else:
                return rand_argmax(utilities)

        posterior_estimator = self.posterior_estimator.fit(X, y)

        if assume_linear:
            return self._query_assume_lin(X_cand, reg, E, X, y,
                                          posterior_estimator,
                                          utilities, return_utilities)

        else:
            return self._query_not_assume_lin(X_cand, reg, E, X, y,
                                              posterior_estimator,
                                              utilities, return_utilities)

    def _query_assume_lin(self, X_cand, reg, E, X, y, posterior_estimator,
                          utilities, return_utilities):

        My_cand, Var_cand = posterior_estimator.estimate_mu_cov(X_cand)

        for idx, (x_c, my_c, var_c) in enumerate(zip(X_cand, My_cand, Var_cand)):
            E = np.array([x_c])
            sigma_v = (var_c.diagonal())**(1/2)
            perf = self.x_perf_assume_linear(reg, E, x_c, my_c, sigma_v, X, y)
            utilities[idx] = perf

        if return_utilities:
            return rand_argmax(utilities), utilities
        else:
            return rand_argmax(utilities)

    def _query_not_assume_lin(self, X_cand, reg, E, X, y, posterior_estimator,
                              utilities, return_utilities):

        n_rvs = self.n_monte_carlo_samples
        Y_cand = posterior_estimator.estimate_random_variates(X_cand, n_rvs)

        for idx_x_c, x_c in enumerate(X_cand):
            utilities_given_y_c = np.zeros(shape=self.n_monte_carlo_samples)
            for idx_y_c, y_c in enumerate(Y_cand[idx_x_c]):
                utilities_given_y_c[idx_y_c] = self.x_perf(reg, E, x_c, y_c, X, y)
            utilities[idx_x_c] = np.average(utilities_given_y_c)

        if return_utilities:
            return rand_argmax(utilities), utilities
        else:
            return rand_argmax(utilities)

    def x_perf(self, reg, E, x_c, y_c, X, y):

        X_new = np.append(X, [x_c], axis=0)
        y_new = np.append(y, [y_c], axis=0)

        reg_old = reg
        reg_new = deepcopy(reg)
        reg_new.fit(X_new, y_new)

        my = self.posterior_estimator.fit(X_new, y_new).predict(E)

        y_pred_old = reg_old.predict(E)
        y_pred_new = reg_new.predict(E)

        error_old = np.sum((y_pred_old - my) ** 2)
        error_new = np.sum((y_pred_new - my) ** 2)

        return np.average(error_old - error_new)

    def x_perf_assume_linear(self, reg, E, x_c, my_c, std_c, X, y):

        modes = [-std_c, 0, std_c]

        X_new = np.append(X, [x_c], axis=0)
        y_new_s = [np.append(y, [my_c + alpha], axis=0) for alpha in modes]

        reg_old = reg
        reg_new_s = [deepcopy(reg) for _ in modes]

        for reg_new, y_new in zip(reg_new_s, y_new_s):
            reg_new.fit(X_new, y_new)

        my_s = [self.posterior_estimator.fit(X_new, y_new).predict(E)
                for y_new in y_new_s]

        y_pred_old = reg_old.predict(E)
        y_pred_new_s = [reg_new.predict(E) for reg_new in reg_new_s]

        error_old = np.sum((y_pred_old - my_s[1]) ** 2)
        error_new = np.sum((y_pred_new_s[1] - my_s[1]) ** 2)

        var_new = 1/2*(np.sum(((my_s[2] - my_s[1])
                               - (y_pred_new_s[2] - y_pred_new_s[1]))**2)
                       + np.sum(((my_s[0] - my_s[1])
                                 - (y_pred_new_s[0] - y_pred_new_s[1]))**2))

        var_old = 1/2*(np.sum((my_s[2] - my_s[1])**2)
                       + np.sum((my_s[0] - my_s[1])**2))

        return np.average((error_old - error_new) + (var_old - var_new))