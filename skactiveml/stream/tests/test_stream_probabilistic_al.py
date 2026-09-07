from copy import deepcopy
import unittest

import numpy as np
from sklearn.naive_bayes import GaussianNB
from skactiveml.utils import MISSING_LABEL
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.stream import StreamProbabilisticAL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorStreamQueryStrategy,
)
from sklearn.svm import SVC


class TestStreamProbabilisticAL(
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
            qs_class=StreamProbabilisticAL,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )
        self.update_params = {
            "candidates": np.array([[1, 2]]),
            "queried_indices": [],
            "budget_manager_param_dict": {"utilities": 0.0},
        }

    def test_query_preserves_constructor_parameters(self):
        super().test_query_preserves_constructor_parameters()
        for metric_dict in [None, {}, {"gamma": 0.5}]:
            super().test_query_preserves_constructor_parameters(
                init_param_overrides={
                    "metric": "rbf",
                    "metric_dict": metric_dict,
                }
            )

    def test_query_metric_defaults_and_kernel_change(self):
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 1, np.nan, np.nan])
        clf = ParzenWindowClassifier(classes=[0, 1]).fit(X, y)
        query_kwargs = dict(
            X=X,
            y=y,
            clf=clf,
            fit_clf=False,
            return_utilities=True,
            candidates=X[2:],
        )
        for metric_dict in [None, {}, {"gamma": 0.5}]:
            with self.subTest(metric_dict=metric_dict):
                qs = StreamProbabilisticAL(
                    metric="rbf", metric_dict=metric_dict, random_state=0
                )
                reference = StreamProbabilisticAL(
                    metric="rbf",
                    random_state=0,
                    metric_dict=(
                        {"gamma": "mean"}
                        if metric_dict is None
                        else deepcopy(metric_dict)
                    ),
                ).query(**query_kwargs)
                for _ in range(2):
                    indices, utilities = qs.query(**query_kwargs)
                    np.testing.assert_array_equal(indices, reference[0])
                    np.testing.assert_allclose(utilities, reference[1])
                if metric_dict is None:
                    qs.set_params(metric="linear")
                    indices, utilities = qs.query(**query_kwargs)
                    reference = StreamProbabilisticAL(
                        metric="linear", random_state=0
                    ).query(**query_kwargs)
                    np.testing.assert_array_equal(indices, reference[0])
                    np.testing.assert_allclose(utilities, reference[1])

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), TypeError),
            (ParzenWindowClassifier(), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = [0, 6]
        expected_utilities = [
            4.3857739e-02,
            9.2920475e-04,
            2.6554099e-02,
            2.5520705e-02,
            1.8695656e-04,
            2.1625776e-03,
            1.0521981e-01,
            1.2177500e-02,
            5.8971376e-02,
            2.3056263e-03,
            7.9819082e-04,
            9.1568692e-05,
            1.7785138e-03,
            1.1380995e-04,
            2.1395855e-03,
            2.9674745e-04,
        ]
        budget_manager_param_dict = {
            "utilities": [
                7.00998185e-05,
                8.88936175e-05,
                4.31959419e-03,
                5.26122086e-03,
            ]
        }
        return super().test_query(
            expected_output, expected_utilities, budget_manager_param_dict
        )

    def test_query_param_utility_weight(self):
        test_cases = []
        random_state = np.random.RandomState(0)
        utility_weight = random_state.rand(
            len(self.query_default_params_clf["candidates"])
        )
        test_cases += [
            ([1, 1], ValueError),
            ("string", ValueError),
            (["string", "string"], ValueError),
            (0.5, TypeError),
            (utility_weight, None),
        ]
        self._test_param("query", "utility_weight", test_cases)

    def test_init_param_prior(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, ValueError),
            (-1.0, ValueError),
            (0.5, None),
        ]
        self._test_param("init", "prior", test_cases)

    def test_init_param_m_max(self):
        test_cases = []
        test_cases += [
            (0.1, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (2, None),
        ]
        self._test_param("init", "m_max", test_cases)

    def test_init_param_metric(self):
        test_cases = []
        test_cases += [("rbf", None), ("", ValueError), (0.5, ValueError)]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self):
        test_cases = []
        test_cases += [
            ({"gamma": "mean"}, None),
            ({"gamma": 0.1}, None),
            (["gamma"], TypeError),
            ({"test": 0}, TypeError),
        ]
        replace_init_params = {"metric": "rbf"}
        self._test_param(
            "init",
            "metric_dict",
            test_cases,
            replace_init_params=replace_init_params,
        )
