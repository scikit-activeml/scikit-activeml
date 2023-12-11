import unittest

import numpy as np
from skactiveml.utils import MISSING_LABEL
from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.stream import (
    FixedUncertainty,
    VariableUncertainty,
    Split,
    RandomVariableUncertainty,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorStreamQueryStrategy,
)


class TestFixedUncertainty(
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
            qs_class=FixedUncertainty,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB()), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

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


class TestSplit(TemplateSingleAnnotatorStreamQueryStrategy, unittest.TestCase):
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
            qs_class=Split,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB()), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = [4, 5, 6, 7, 14]
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


class TestVariableUncertainty(
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
            qs_class=VariableUncertainty,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB()), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = [4, 5, 6, 7, 8, 14]
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


class TestRandomVariableUncertainty(
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
            qs_class=RandomVariableUncertainty,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB()), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = [0, 1, 2, 4, 5, 6, 7, 12]
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
