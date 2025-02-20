import unittest

import numpy as np
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
            (0, ValueError),
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
        replace_init_params = {"dist_func": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )
        replace_init_params = {"dist_func": pairwise_distances}
        test_cases = [(np.array([[1, 2]]), None)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_init_param_dist_func_dict(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (["func"], TypeError),
            ({"metric": "manhattan"}, None),
        ]
        self._test_param("init", "dist_func_dict", test_cases)
        replace_init_params = {"dist_func_dict": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

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
        expected_output = [1, 4, 5, 6, 7, 9, 10, 11, 15]
        expected_utilities = [
            1.6358911e-04,
            2.4108488e-05,
            1.7458633e-08,
            1.2106466e-03,
            1.1320472e-03,
            1.7989020e-02,
            1.5713135e-01,
            3.4409789e-02,
            9.2470118e-02,
            1.9605512e-02,
            6.9009195e-03,
            5.9577877e-05,
            1.8402160e-03,
            5.2106541e-05,
            2.8001754e-03,
            2.5658824e-03,
        ]
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
        expected_output = [6]
        expected_utilities = [
            1.6358911e-04,
            2.4108488e-05,
            1.7458633e-08,
            1.2106466e-03,
            1.1320472e-03,
            1.7989020e-02,
            1.5713135e-01,
            3.4409789e-02,
            9.2470118e-02,
            1.9605512e-02,
            6.9009195e-03,
            5.9577877e-05,
            1.8402160e-03,
            5.2106541e-05,
            2.8001754e-03,
            2.5658824e-03,
        ]
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
        expected_output = [1, 4, 5, 6, 7, 9, 10, 11, 15]
        expected_utilities = [
            1.6358911e-04,
            2.4108488e-05,
            1.7458633e-08,
            1.2106466e-03,
            1.1320472e-03,
            1.7989020e-02,
            1.5713135e-01,
            3.4409789e-02,
            9.2470118e-02,
            1.9605512e-02,
            6.9009195e-03,
            5.9577877e-05,
            1.8402160e-03,
            5.2106541e-05,
            2.8001754e-03,
            2.5658824e-03,
        ]
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
        expected_utilities = [
            1.6358911e-04,
            2.4108488e-05,
            1.7458633e-08,
            1.2106466e-03,
            1.1320472e-03,
            1.7989020e-02,
            1.5713135e-01,
            3.4409789e-02,
            9.2470118e-02,
            1.9605512e-02,
            6.9009195e-03,
            5.9577877e-05,
            1.8402160e-03,
            5.2106541e-05,
            2.8001754e-03,
            2.5658824e-03,
        ]
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
            init_default_params={
                "force_full_budget": True,
                "classes": self.classes,
            },
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = []
        expected_utilities = [
            1.6358911e-04,
            2.4108488e-05,
            1.7458633e-08,
            1.2106466e-03,
            1.1320472e-03,
            1.7989020e-02,
            1.5713135e-01,
            3.4409789e-02,
            9.2470118e-02,
            1.9605512e-02,
            6.9009195e-03,
            5.9577877e-05,
            1.8402160e-03,
            5.2106541e-05,
            2.8001754e-03,
            2.5658824e-03,
        ]
        return super().test_query(expected_output, expected_utilities)

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, TypeError),
            (CognitiveDualQueryStrategyFixUn, TypeError),
        ]
        self._test_param("init", "classes", test_cases)
        self._test_param("init", "classes", [([0, 1], None)])
        self._test_param(
            "init",
            "classes",
            [(["0", "1"], None)],
            {},
            {"y": ["0", "1", "none", "none"]},
        )


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
            (0, ValueError),
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
        replace_init_params = {"dist_func": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )
        replace_init_params = {"dist_func": pairwise_distances}
        test_cases = [(np.array([[1, 2]]), None)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_init_param_dist_func_dict(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (["func"], TypeError),
            ({"metric": "manhattan"}, None),
        ]
        self._test_param("init", "dist_func_dict", test_cases)
        replace_init_params = {"dist_func_dict": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB(), classes=[0, 1]), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = []
        expected_utilities = [
            3.2717822e-04,
            4.8216976e-05,
            3.4917266e-08,
            2.4212931e-03,
            2.2640944e-03,
            3.5978039e-02,
            3.1426270e-01,
            6.8819578e-02,
            1.8494024e-01,
            3.9211023e-02,
            1.3801839e-02,
            1.1915575e-04,
            3.6804321e-03,
            1.0421308e-04,
            5.6003508e-03,
            5.1317648e-03,
        ]
        return super().test_query(expected_output, expected_utilities)
