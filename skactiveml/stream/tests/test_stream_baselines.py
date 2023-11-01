import unittest

import numpy as np
from skactiveml.stream import PeriodicSampling, StreamRandomSampling
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorStreamQueryStrategy,
)


class TestStreamRandomSampling(
    TemplateSingleAnnotatorStreamQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
        }
        super().setUp(
            qs_class=StreamRandomSampling,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_allow_exceeding_budget(self):
        test_cases = []
        test_cases += [(0, TypeError), ("", TypeError), (True, None)]
        self._test_param("init", "allow_exceeding_budget", test_cases)

    def test_query(self):
        expected_output = np.array([])
        expected_utilities = np.array([0.6458941])
        return super().test_query(expected_output, expected_utilities)


class TestPeriodicSampling(
    TemplateSingleAnnotatorStreamQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
        }
        super().setUp(
            qs_class=PeriodicSampling,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = np.array([])
        expected_utilities = np.array([0.0])
        return super().test_query(expected_output, expected_utilities)
