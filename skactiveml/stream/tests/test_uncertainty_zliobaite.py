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
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0, classes=self.classes).fit(
            X, y
        )
        query_default_params_clf = {
            "candidates": np.array([[1, 2]]),
            "X": X,
            "clf": clf,
            "y": y,
        }
        init_default_params = {
            "classes": self.classes,
        }
        super().setUp(
            qs_class=FixedUncertainty,
            init_default_params=init_default_params,
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, TypeError),
            (FixedUncertainty, TypeError),
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

        qs_2 = FixedUncertainty(classes=[0, 1], random_state=0)
        queried_indices_2, utilities_2 = qs_2.query(
            candidates=[[1, 2]],
            X=[[0, 2], [2, 2]],
            y=[0, 1],
            clf=ParzenWindowClassifier(classes=[0, 1], random_state=0),
            fit_clf=True,
            return_utilities=True,
        )

        qs_3 = FixedUncertainty(classes=[0, 1, 2], random_state=0)
        queried_indices_3, utilities_3 = qs_3.query(
            candidates=[[1, 2]],
            X=[[0, 2], [2, 2]],
            y=[0, 1],
            clf=ParzenWindowClassifier(classes=[0, 1, 2], random_state=0),
            fit_clf=True,
            return_utilities=True,
        )
        # The probabilities should be 50/50 for class 0 and 1. Thus, the
        # confidence of the classifier is 0.5, which is sufficient to query the
        # label in a 2-class setting but not in a 3-class setting
        self.assertEqual(len(queried_indices_2), 1)
        self.assertEqual(len(queried_indices_3), 0)

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
