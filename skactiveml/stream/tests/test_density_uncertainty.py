import unittest

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from skactiveml.utils import MISSING_LABEL
from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.stream import (
    StreamDensityBasedAL,
    CognitiveDualQueryStrategy,
    CognitiveDualQueryStrategyRan,
    CognitiveDualQueryStrategyRanVarUn,
    CognitiveDualQueryStrategyVarUn,
    CognitiveDualQueryStrategyFixUn,
)

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorStreamQueryStrategy,
)


class TemplateCognitiveDualQueryStrategy(
    TemplateSingleAnnotatorStreamQueryStrategy
):
    def test_init_param_density_threshold(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, TypeError),
            (-1, ValueError),
            (1, None),
        ]
        self._test_param("init", "density_threshold", test_cases)

    def test_init_param_cognition_window_size(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, TypeError),
            (-1, ValueError),
            (10, None),
        ]
        self._test_param("init", "cognition_window_size", test_cases)

    def test_init_param_dist_func(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (pairwise_distances, None),
            (None, None),
        ]
        self._test_param("init", "dist_func", test_cases)

    def test_init_param_dist_func_dict(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (["func"], TypeError),
            ({"metric": "manhattan"}, None),
        ]
        self._test_param("init", "dist_func_dict", test_cases)

    def test_init_param_force_full_budget(self):
        test_cases = []
        test_cases += [("string", TypeError), (True, None), (False, None)]
        self._test_param("init", "force_full_budget", test_cases)

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB()), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)


class TestCognitiveDualQueryStrategy(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 0, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        super().setUp(
            qs_class=CognitiveDualQueryStrategy,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = [0]
        expected_utilities = [0.3841551]
        return super().test_query(expected_output, expected_utilities)


class TestCognitiveDualQueryStrategyVarUn(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 0, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        super().setUp(
            qs_class=CognitiveDualQueryStrategyVarUn,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = [0]
        expected_utilities = [0.3841551]
        return super().test_query(expected_output, expected_utilities)


class TestCognitiveDualQueryStrategyRanVarUn(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 0, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        super().setUp(
            qs_class=CognitiveDualQueryStrategyRanVarUn,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = [0]
        expected_utilities = [0.3841551]
        return super().test_query(expected_output, expected_utilities)


class TestCognitiveDualQueryStrategyRan(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 0, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        super().setUp(
            qs_class=CognitiveDualQueryStrategyRan,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = []
        expected_utilities = [0.3841551]
        return super().test_query(expected_output, expected_utilities)


class TestCognitiveDualQueryStrategyFixUn(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 0, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        super().setUp(
            qs_class=CognitiveDualQueryStrategyFixUn,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = []
        expected_utilities = [0.3841551]
        return super().test_query(expected_output, expected_utilities)


class TestStreamDensityBasedAL(
    TemplateSingleAnnotatorStreamQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 0, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        super().setUp(
            qs_class=StreamDensityBasedAL,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_window_size(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, TypeError),
            (-1, ValueError),
            (100, None),
        ]
        self._test_param("init", "window_size", test_cases)

    def test_init_param_dist_func(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (pairwise_distances, None),
            (None, None),
        ]
        self._test_param("init", "dist_func", test_cases)

    def test_init_param_dist_func_dict(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (["func"], TypeError),
            ({"metric": "manhattan"}, None),
        ]
        self._test_param("init", "dist_func_dict", test_cases)

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB(), classes=[0, 1]), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = []
        expected_utilities = [0.7683102]
        return super().test_query(expected_output, expected_utilities)
