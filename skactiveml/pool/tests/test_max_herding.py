import unittest

import numpy as np
from sklearn.datasets import make_blobs
from skactiveml.pool import MaxHerding
from skactiveml.utils import is_unlabeled
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestMaxHerding(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):

    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=500, centers=5, random_state=0, shuffle=True
        )
        self.y[:450] = -1
        query_default_params = {
            "X": self.X,
            "y": self.y,
        }
        super().setUp(
            qs_class=MaxHerding,
            init_default_params={"missing_label": -1},
            query_default_params_clf=query_default_params,
        )

    def test_init_param_metric(self):
        test_cases = [
            ("rbf", None),
            (lambda x, y: ((x - y) ** 2).sum(), None),
            (None, ValueError),
            ([], ValueError),
        ]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self):
        test_cases = [
            ("gamma", TypeError),
            ([], TypeError),
            ({"n_jobs": -1, "gamma": 2}, None),
        ]
        self._test_param("init", "metric_dict", test_cases)

    def test_init_param_normalize_samples(self):
        test_cases = [
            (False, None),
            (True, None),
            (0, TypeError),
            (1, TypeError),
            ("Test", TypeError),
        ]
        self._test_param("init", "normalize_samples", test_cases)

    def test_query(self):
        # All utilities are in [0, 1] or np.nan for 'rbf' as kernel.
        max_herding_1 = MaxHerding(random_state=0, missing_label=-1)
        _, utilities_1 = max_herding_1.query(
            self.X, self.y, batch_size=2, return_utilities=True
        )
        self.assertFalse((utilities_1 < 0).any())
        self.assertFalse((utilities_1 > 1).any())

        # Using "linear" kernel leads to different utilities.
        max_herding_2 = MaxHerding(
            random_state=0, metric="linear", missing_label=-1
        )
        _, utilities_2 = max_herding_2.query(
            self.X, self.y, batch_size=2, return_utilities=True
        )
        self.assertTrue(utilities_1[0, 0] - utilities_2[0, 0])

        # Not normalizing samples leads to different utilities.
        max_herding_3 = MaxHerding(
            random_state=0, normalize_samples=False, missing_label=-1
        )
        _, utilities_3 = max_herding_3.query(
            self.X, self.y, batch_size=2, return_utilities=True
        )
        self.assertTrue(utilities_1[0, 0] - utilities_3[0, 0])

        # If all samples are identical, their utilities are also identical.
        X_3 = np.ones((10, 2))
        y_3 = np.full((10,), -1)
        _, utilities_4 = max_herding_1.query(
            X_3, y_3, batch_size=1, return_utilities=True
        )
        self.assertTrue(np.unique(utilities_4)[0] == 1.0)

        # Candidates are given as indices.
        candidates = np.arange(0, 5)
        _, utilities_5 = max_herding_1.query(
            self.X,
            self.y,
            batch_size=1,
            candidates=candidates,
            return_utilities=True,
        )
        self.assertTrue(np.isnan(utilities_5[0, 5:]).all())
        self.assertFalse(np.isnan(utilities_5[0, :5]).any())

        # Candidates are given as new samples.
        is_ulbld = is_unlabeled(self.y, missing_label=-1)
        _, utilities_6 = max_herding_1.query(
            self.X,
            self.y,
            candidates=self.X[is_ulbld],
            batch_size=2,
            return_utilities=True,
        )
        np.testing.assert_array_equal(
            np.unique(utilities_1), np.unique(utilities_6)
        )
