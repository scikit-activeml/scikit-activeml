import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1, make_regression
import matplotlib.pyplot as plt
from scipy.stats import t

from skactiveml.base import SkactivemlRegressor
from skactiveml.regressor._wrapper import SklearnRegressor
from skactiveml.utils import MISSING_LABEL


class NWR(SkactivemlRegressor):
    METRICS = list(KERNEL_PARAMS.keys()) + ['precomputed']

    def __init__(self, n_neighbors=None, metric='rbf', metric_dict=None,
                 random_state=None):
        super().__init__(random_state=random_state)
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.metric_dict = {} if metric_dict is None else metric_dict

    def fit(self, X, y, sample_weight=None):
        self.X_ = X.copy()
        self.y_ = y.copy()

        return self

    def predict(self, X):
        K = pairwise_kernels(X, self.X_, metric=self.metric,
                             **self.metric_dict)


        # maximum likelihood
        N = np.sum(K, axis=1)
        mu_ml = K @ self.y_ / N
        return mu_ml