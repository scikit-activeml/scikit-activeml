import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.pool.regression._information_maximization import (
    MutualInformationGainMaximization,
)
from skactiveml.regressor._wrapper import SklearnTargetDistributionRegressor


class TestMIM(unittest.TestCase):
    def setUp(self):
        pass

    def test_query(self):
        mim = MutualInformationGainMaximization(random_state=0)

        cond_est = SklearnTargetDistributionRegressor(
            estimator=GaussianProcessRegressor()
        )

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        cond_est.fit(X, y)

        query_indices = mim.query(
            X, y, candidates=X_cand, cond_est=cond_est, batch_size=2
        )
        print(query_indices)
