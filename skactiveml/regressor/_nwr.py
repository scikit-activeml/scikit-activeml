from sklearn.metrics.pairwise import KERNEL_PARAMS

from skactiveml.base import SkactivemlRegressor
from skactiveml.regressor.estimator._ngke import NormalGammaKernelEstimator


class NWR(SkactivemlRegressor):
    METRICS = list(KERNEL_PARAMS.keys()) + ['precomputed']

    def __init__(self, metric='rbf', metric_dict=None, random_state=None):
        super().__init__(random_state=random_state)
        self.ngke = NormalGammaKernelEstimator(metric=metric,
                                               metric_dict=metric_dict,
                                               lmbda_0=0, random_state=None)

    def fit(self, X, y, sample_weight=None):
        self.ngke.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        return self.ngke.predict(X)