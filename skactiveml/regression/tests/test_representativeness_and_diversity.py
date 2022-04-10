import unittest

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.regression._expected_model_variance import (
    ExpectedModelVarianceMinimization,
)
from skactiveml.regression._mutual_information_maximization import (
    MutualInformationGainMaximization,
)
from skactiveml.regression._representativeness_and_diversity import (
    RepresentativenessAndDiversity,
)
from skactiveml.regressor._wrapper import SklearnConditionalEstimator
from skactiveml.utils import MISSING_LABEL


class TestMIM(unittest.TestCase):
    def setUp(self):
        pass

    def test_query(self):
        qs = RepresentativenessAndDiversity()

        X_cand = np.array([[1, 0], [0, 0], [0, 1], [-10, 1], [10, -10]])
        X = np.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        y = np.array([0, 1])

        query_indices = qs.query(X, y, candidates=X_cand, batch_size=2)
        print(query_indices)

    def test_query_2(self):
        qs = RepresentativenessAndDiversity()

        X = np.array([[1, 2], [3, 4], [0, 0], [0, 1], [-10, 1]])
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL])

        query_indices = qs.query(X, y, batch_size=2)
        print(query_indices)

    def test_query_3(self):
        qs = RepresentativenessAndDiversity()

        X = np.array([[1, 2], [3, 4], [0, 0], [0, 1], [-10, 1]])
        y = np.array([1.0] + [MISSING_LABEL] * 4)

        query_indices = qs.query(X, y, batch_size=2)
        print(query_indices)

    def test_query_4(self):
        qs = RepresentativenessAndDiversity(qs=ExpectedModelVarianceMinimization())
