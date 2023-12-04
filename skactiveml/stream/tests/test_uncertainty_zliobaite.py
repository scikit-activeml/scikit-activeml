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
        expected_output = [4, 9, 14, 19]
        expected_utilities = [
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
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
        expected_output = [0, 1, 2, 3, 4, 5, 11]
        expected_utilities = [
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
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
        expected_output = [0, 1, 2, 3, 4, 5, 11]
        expected_utilities = [
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
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
        expected_output = [0, 1, 2, 3, 5, 9, 12]
        expected_utilities = [
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
            0.3841551,
            0.4388273,
            0.3902932,
            0.1933426,
            0.4901074,
        ]
        return super().test_query(expected_output, expected_utilities)
