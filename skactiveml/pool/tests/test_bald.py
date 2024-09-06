import unittest
from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    BaggingClassifier,
)
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import (
    GaussianProcessRegressor,
    GaussianProcessClassifier,
)

from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier

from skactiveml.pool._bald import (
    _GeneralBALD,
    batch_bald,
    BatchBALD,
    GreedyBALD,
)
from skactiveml.regressor import NICKernelRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestGeneralBALD(
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
            qs_class=_GeneralBALD,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_n_MC_samples(self):
        test_cases = [
            (0, ValueError),
            (1.2, TypeError),
            (1, None),
            (None, None),
        ]
        self._test_param("init", "n_MC_samples", test_cases)

    def test_init_param_greedy_selection(self):
        test_cases = [
            (0, TypeError),
            (1.2, TypeError),
            (1, TypeError),
            ("1", TypeError),
            (False, None),
            (True, None),
        ]
        self._test_param("init", "greedy_selection", test_cases)
        self.assertTrue(GreedyBALD().greedy_selection)
        self.assertFalse(BatchBALD().greedy_selection)

    def test_init_param_eps(self):
        test_cases = [
            (0, ValueError),
            (1e-3, None),
            (0.1, None),
            ("1", TypeError),
            (1, ValueError),
        ]
        self._test_param(
            "init",
            "eps",
            test_cases,
        )

    def test_init_param_sample_predictions_method_name(self):
        # Fails as the default ensemble from `setup` does not support sampling.
        test_cases = [
            (0, TypeError),
            (0.1, TypeError),
            ("Test", ValueError),
        ]
        self._test_param("init", "sample_predictions_method_name", test_cases)
        test_cases = [
            ("predict_proba", ValueError),
        ]
        self._test_param(
            "init",
            "sample_predictions_method_name",
            test_cases,
            replace_query_params={
                "ensemble": [
                    ParzenWindowClassifier(),
                    ParzenWindowClassifier(),
                ]
            },
        )
        test_cases = [
            ("sample_proba", None),
        ]
        self._test_param(
            "init",
            "sample_predictions_method_name",
            test_cases,
            replace_query_params={
                "ensemble": ParzenWindowClassifier(),
                "fit_ensemble": True,
            },
        )

    def test_init_param_sample_predictions_dict(self):
        test_cases = [
            (None, None),
            ({}, None),
            ({"n_samples": 1000}, None),
            ("Test", ValueError),
            ({"Test": 2}, TypeError),
        ]
        self._test_param(
            "init",
            "sample_predictions_dict",
            test_cases,
            replace_init_params={
                "sample_predictions_method_name": "sample_proba",
            },
            replace_query_params={
                "ensemble": ParzenWindowClassifier(),
                "fit_ensemble": True,
            },
        )
        test_cases = [
            (None, None),
            ({}, ValueError),
            ({"n_samples": 1000}, ValueError),
            ("Test", ValueError),
            ({"Test": 2}, ValueError),
        ]
        self._test_param(
            "init",
            "sample_predictions_dict",
            test_cases,
            replace_init_params={
                "sample_predictions_method_name": None,
            },
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
            replace_query_params={"fit_ensemble": False},
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
        gpc = ParzenWindowClassifier(classes=self.classes)
        ensemble_bagging = SklearnClassifier(
            estimator=BaggingClassifier(estimator=gpc, random_state=42),
            classes=self.classes,
        )
        ensemble_array_clf = [
            ParzenWindowClassifier(classes=self.classes, random_state=41),
            ParzenWindowClassifier(classes=self.classes, random_state=42),
        ]
        ensemble_list = [
            self.ensemble_clf,
            ensemble_bagging,
            ensemble_array_clf,
        ]
        for ensemble in ensemble_list:
            with self.subTest(init_labels=ensemble):
                query_params = deepcopy(self.query_default_params_clf)
                batch_size = 2
                query_params["batch_size"] = batch_size
                query_params["ensemble"] = ensemble
                query_params["return_utilities"] = True
                for greedy_selection in [False, True]:
                    for candidates in [None, [2, 3]]:
                        # query_params["candidates"] = candidates
                        qs = self.qs_class(
                            greedy_selection=greedy_selection, random_state=42
                        )
                        np.testing.assert_equal(
                            qs.query(**query_params)[1],
                            qs.query(**query_params)[1],
                        )
                        idx, u = qs.query(**query_params)
                        self.assertEqual(len(idx), batch_size)
                        self.assertEqual(len(u), batch_size)


class Testbatch_bald(unittest.TestCase):
    def setUp(self):
        p = np.random.rand(10, 100, 1)
        self.default_params = {
            "probas": np.append(p, 1 - p, axis=2),
            "batch_size": 1,
            "random_state": 0,
        }

    def test_param_probas(self):
        test_cases = [
            (None, AttributeError),
            (np.random.rand(10, 100), ValueError),
            (np.random.rand(10, 100, 3), None),
            (np.random.rand(10, 100, 3, 2), ValueError),
        ]
        self._test_param(batch_bald, "probas", test_cases)

    def test_param_batch_size(self):
        test_cases = [(0, ValueError), (1.2, TypeError), (1, None)]
        self._test_param(batch_bald, "batch_size", test_cases)

    def test_param_n_MC_samples(self):
        test_cases = [
            (0, ValueError),
            (1.2, TypeError),
            (1, None),
            (None, None),
        ]
        self._test_param(batch_bald, "n_MC_samples", test_cases)

    def test_param_eps(self):
        test_cases = [
            ("0", TypeError),
            (1, ValueError),
            (-1, ValueError),
            (-0.1, ValueError),
            (0.001, None),
            (0, ValueError),
            (0.1, None),
        ]
        self._test_param(batch_bald, "eps", test_cases)

    def test_param_random_state(self):
        test_cases = [(np.nan, ValueError), ("state", ValueError), (1, None)]
        self._test_param(batch_bald, "random_state", test_cases)

    def test_batch_bald(self):
        # test BALD and BatchBALD
        probas = np.random.rand(10, 100, 5)
        np.testing.assert_equal(_bald(probas), _bald(probas))
        np.testing.assert_allclose(
            _bald(probas), batch_bald(probas, batch_size=1)[0], rtol=1e-6
        )
        np.testing.assert_equal(
            batch_bald(probas, batch_size=1), batch_bald(probas, batch_size=1)
        )

        batch_size = 20
        n_estimators = 10
        n_classes = 2
        n_samples = 200
        random_state = np.random.RandomState(0)
        probas = random_state.random(
            n_classes * n_samples * n_estimators
        ).reshape((n_estimators, n_samples, n_classes))
        probas /= np.sum(probas, keepdims=True, axis=-1)

        # utils3 = get_batchbald_batch(
        #     log_probs_N_K_C=torch.Tensor(np.log(probas.swapaxes(0, 1))),
        #     batch_size=batch_size, num_samples=n_estimators,
        #     random_state=np.random.RandomState(0))
        # expected_max_utilities = np.array(utils3.scores)
        expected_max_utilities = np.array(
            [
                0.26082319,
                0.51374567,
                0.73904878,
                0.93539158,
                1.07305285,
                1.27963334,
                1.32600832,
                1.32793891,
                1.77275521,
                1.74883515,
                1.81012598,
                1.91902003,
                1.90019259,
                2.14320686,
                1.91724786,
                2.59798092,
                2.29420766,
                2.14945838,
                3.22228023,
                1.69877866,
            ]
        )

        utils = batch_bald(
            probas,
            batch_size=batch_size,
            n_MC_samples=n_estimators,
            random_state=np.random.RandomState(0),
        )

        np.testing.assert_allclose(
            expected_max_utilities, np.nanmax(utils, axis=1), rtol=1e-6
        )

    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
    ):
        for i, (test_val, err) in enumerate(test_cases):
            with self.subTest(msg="Param", id=i, val=test_val):
                params = deepcopy(self.default_params)
                params[test_param] = test_val

                if err is None:
                    test_func(**params)
                else:
                    self.assertRaises(err, test_func, **params)


def _bald(probas):
    """
    Computes the Bayesian Active Learning by Disagreement (BALD) score for
    each sample.

    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.

    Returns
    -------
    scores: np.ndarray, shape (n_samples)
        The BALD-scores.

    References
    ----------
    [1] Houlsby, Neil, et al. "Bayesian active learning for classification and
        preference learning." arXiv preprint arXiv:1112.5745 (2011).
    """
    p_mean = np.mean(probas, axis=0)
    uncertainty = np.nansum(-p_mean * np.log(p_mean), axis=1)
    confident = np.nanmean(np.nansum(-probas * np.log(probas), axis=2), axis=0)
    return uncertainty - confident
