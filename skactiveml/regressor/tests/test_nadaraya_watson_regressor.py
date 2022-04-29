import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from skactiveml.regressor import NadarayaWatsonRegressor
from skactiveml.utils import MISSING_LABEL


class TestNadarayaWatsonRegressor(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.random_state = 0
        self.start_parameter = {
            "metric": "rbf",
            "metric_dict": {"gamma": 10.0},
            "missing_label": MISSING_LABEL,
            "random_state": self.random_state,
        }

    def test_fit(self):
        reg = NadarayaWatsonRegressor(**self.start_parameter)
        self.assertRaises(NotFittedError, reg.predict, self.X)
        X = np.zeros((3, 1))
        y = np.arange(3)
        reg.fit(X, y)
        y_pred = reg.predict([[0]])[0]
        self.assertEqual(y_pred, np.average(y))
