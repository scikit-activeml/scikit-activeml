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
        test_cases += [
            (0, TypeError),
            ("", TypeError),
            (True, None),
            (False, None),
        ]
        self._test_param("init", "allow_exceeding_budget", test_cases)

    def test_query(self):
        expected_output = np.array([4, 9])
        expected_utilities = [
            0.4236548,
            0.6458941,
            0.4375872,
            0.891773,
            0.9636628,
            0.3834415,
            0.791725,
            0.5288949,
            0.5680446,
            0.9255966,
            0.0710361,
            0.0871293,
            0.0202184,
            0.8326198,
            0.7781568,
            0.8700121,
        ]
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
        expected_output = np.array([9, 19, 29, 39])
        stream_length = 40
        expected_utilities = np.full(stream_length, fill_value=0.0)
        expected_utilities[expected_output] = 1
        candidates = np.zeros((stream_length, 1))
        queried_indices = np.arange(0, 4)

        super().test_query(
            expected_output,
            expected_utilities,
            candidates=candidates,
            queried_indices=queried_indices,
        )
