import unittest
from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, \
    RandomForestRegressor, BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor, \
    GaussianProcessClassifier

from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier

from skactiveml.pool._bald import BALD, _bald, _batchbald
from skactiveml.regressor import NICKernelRegressor
from skactiveml.tests.template_query_strategy import \
    TemplateSingleAnnotatorPoolQueryStrategy
from skactiveml.utils import MISSING_LABEL


class TestBALD(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        self.ensemble_clf = SklearnClassifier(
            estimator=RandomForestClassifier(random_state=42),
            classes=self.classes,
            random_state=42,
        )
        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            "ensemble": self.ensemble_clf,
            "fit_ensemble": True,
        }
        super().setUp(
            qs_class=BALD,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query_param_ensemble(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, TypeError),
            ("test", TypeError),
            (1, TypeError),
            (ParzenWindowClassifier(classes=self.classes), TypeError),
            (GaussianProcessRegressor(), TypeError),
            (RandomForestRegressor(), TypeError),
            (RandomForestClassifier(), TypeError),
            (
                [GaussianProcessRegressor(), GaussianProcessRegressor()],
                TypeError,
            ),
            (
                [GaussianProcessClassifier(), GaussianProcessRegressor()],
                TypeError,
            ),
            ([NICKernelRegressor(), ParzenWindowClassifier()], TypeError),
            (self.ensemble_clf, None),
            ([ParzenWindowClassifier(), ParzenWindowClassifier()], None),
        ]
        self._test_param("query", "ensemble", test_cases)

        pwc_list = [ParzenWindowClassifier(), ParzenWindowClassifier()]
        test_cases = [(pwc_list, NotFittedError)]
        self._test_param(
            "query",
            "ensemble",
            test_cases,
            replace_query_params={"fit_ensemble": False}
        )

    def test_query_param_y(self, test_cases=None):
        y = self.query_default_params_clf["y"]
        test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
        self._test_param("query", "y", test_cases, exclude_reg=True)

        for ml, classes, t, err in [
            (np.nan, [1.0, 2.0], float, None),
            (0, [1, 2], int, None),
            (None, [1, 2], object, None),
            (None, ["A", "B"], object, None),
            ("", ["A", "B"], str, None),
        ]:
            replace_init_params = {"missing_label": ml}

            ensemble = clone(self.query_default_params_clf["ensemble"])
            ensemble.missing_label = ml
            ensemble.classes = classes
            replace_query_params = {"ensemble": ensemble}

            replace_y = np.full_like(y, ml, dtype=t)
            replace_y[0] = classes[0]
            replace_y[1] = classes[1]
            test_cases = [(replace_y, err)]
            self._test_param(
                "query",
                "y",
                test_cases,
                replace_init_params=replace_init_params,
                replace_query_params=replace_query_params,
                exclude_reg=True,
            )

    def test_query_param_fit_ensemble(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", TypeError), (None, TypeError)]
        self._test_param("query", "fit_ensemble", test_cases)

    def test_query(self):
        # test _BALD and _BatchBald
        probas = np.random.rand(10, 100, 5)
        np.testing.assert_allclose(_bald(probas), _batchbald(probas)[0])

        gpc = ParzenWindowClassifier(classes=self.classes)
        ensemble_bagging = SklearnClassifier(
            estimator=BaggingClassifier(estimator=gpc),
            classes=self.classes,
        )
        ensemble_array_clf = [
            ParzenWindowClassifier(classes=self.classes),
            ParzenWindowClassifier(classes=self.classes),
        ]
        ensemble_list = [
            self.ensemble_clf,
            ensemble_bagging,
            ensemble_array_clf,
        ]
        for ensemble in ensemble_list:
            query_params = deepcopy(self.query_default_params_clf)
            query_params["ensemble"] = ensemble
            query_params["return_utilities"] = True
            qs = self.qs_class()
            idx, u = qs.query(**query_params)
            self.assertEqual(len(idx), 1)
            self.assertEqual(len(u), 1)
