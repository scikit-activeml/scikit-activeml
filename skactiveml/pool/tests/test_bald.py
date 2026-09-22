import unittest
import pickle
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.stats import entropy
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
    _SampledJointEntropy,
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

    def test_sequence_ensembles_preserve_members(self):
        X = np.arange(4, dtype=float).reshape(-1, 1)
        y = np.array([0, 1, np.nan, np.nan])
        for strategy_class in [GreedyBALD, BatchBALD]:
            for fit_ensemble in [True, False]:
                for prefitted in [False, True]:
                    if not fit_ensemble and not prefitted:
                        continue
                    members = [
                        ParzenWindowClassifier(classes=[0, 1]),
                        ParzenWindowClassifier(classes=[0, 1]),
                    ]
                    if prefitted:
                        for member in members:
                            member.fit(X, [1, 0, np.nan, np.nan])
                    before = [pickle.dumps(member) for member in members]
                    results = []
                    for sequence in [tuple, list]:
                        with self.subTest(
                            strategy=strategy_class.__name__,
                            fit_ensemble=fit_ensemble,
                            prefitted=prefitted,
                            sequence=sequence,
                        ):
                            results.append(
                                strategy_class(random_state=0).query(
                                    X,
                                    y,
                                    ensemble=sequence(members),
                                    fit_ensemble=fit_ensemble,
                                    batch_size=2,
                                    return_utilities=True,
                                )
                            )
                            self.assertEqual(
                                before,
                                [pickle.dumps(member) for member in members],
                            )
                    np.testing.assert_array_equal(results[0][0], results[1][0])
                    np.testing.assert_allclose(results[0][1], results[1][1])

    def test_init_param_n_MC_samples(self):
        test_cases = [
            (0, ValueError),
            (1.2, TypeError),
            (1, None),
            (None, None),
        ]
        self._test_param("init", "n_MC_samples", test_cases)

    def test_greedy_n_MC_samples_is_deprecated(self):
        query_params = deepcopy(self.query_default_params_clf)
        query_params.update(batch_size=2, return_utilities=True)
        expected_indices, expected_utilities = GreedyBALD(
            random_state=0
        ).query(**query_params)
        for n_MC_samples in (None, 1, 10):
            strategy = GreedyBALD(n_MC_samples=n_MC_samples, random_state=0)
            strategies = {
                "constructor": strategy,
                "set_params": GreedyBALD(random_state=0).set_params(
                    n_MC_samples=n_MC_samples
                ),
                "clone": clone(strategy),
            }
            for method, strategy in strategies.items():
                with self.subTest(n_MC_samples=n_MC_samples, method=method):
                    self.assertEqual(
                        strategy.get_params()["n_MC_samples"], n_MC_samples
                    )
                    with self.assertWarnsRegex(
                        FutureWarning, "n_MC_samples.*deprecated.*GreedyBALD"
                    ):
                        indices, utilities = strategy.query(**query_params)
                    np.testing.assert_array_equal(indices, expected_indices)
                    np.testing.assert_array_equal(
                        utilities, expected_utilities
                    )

    def test_default_greedy_and_batch_bald_do_not_warn(self):
        query_params = deepcopy(self.query_default_params_clf)
        query_params.update(batch_size=2, return_utilities=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            strategies = [GreedyBALD(random_state=0), clone(GreedyBALD())]
            strategies.extend(
                BatchBALD(n_MC_samples=n_MC_samples, random_state=0)
                for n_MC_samples in (None, 1, 10)
            )
            for strategy in strategies:
                with self.subTest(strategy=strategy):
                    strategy.query(**query_params)

    def test_tied_batches_match_the_returned_utility_history(self):
        X = np.arange(6.0).reshape(-1, 1)
        y = np.full(len(X), np.nan)
        members = [ParzenWindowClassifier(classes=[0, 1]) for _ in range(3)]
        subset = np.array([4, 1, 5, 2])
        for candidates, seed, n_MC_samples in product(
            (None, subset, X[subset] + 0.5), (0, 2), (None, 1)
        ):
            with self.subTest(
                candidates=candidates, seed=seed, n_MC_samples=n_MC_samples
            ):
                strategy = BatchBALD(
                    random_state=seed, n_MC_samples=n_MC_samples
                )
                indices, utilities = strategy.query(
                    X,
                    y,
                    members,
                    candidates=candidates,
                    batch_size=3,
                    return_utilities=True,
                )
                self.assertEqual(indices.shape, (3,))
                self.assertEqual(len(np.unique(indices)), 3)
                if candidates is None:
                    eligible = np.arange(len(X))
                elif candidates.ndim == 1:
                    eligible = candidates
                else:
                    eligible = np.arange(len(candidates))
                for step, index in enumerate(indices):
                    mask = np.ones(utilities.shape[1], dtype=bool)
                    mask[eligible] = False
                    mask[indices[:step]] = True
                    np.testing.assert_array_equal(
                        np.isnan(utilities[step]), mask
                    )
                    self.assertTrue(
                        np.all(np.isfinite(utilities[step, ~mask]))
                    )
                    self.assertEqual(
                        utilities[step, index], np.nanmax(utilities[step])
                    )
                np.testing.assert_array_equal(
                    indices,
                    strategy.query(
                        X, y, members, candidates=candidates, batch_size=3
                    ),
                )

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

    def test_target_capabilities_are_classification_only(self):
        expected = frozenset(
            {("classification", "single-output", "single-annotator")}
        )
        for strategy in [_GeneralBALD(), BatchBALD(), GreedyBALD()]:
            with self.subTest(strategy=type(strategy).__name__):
                self.assertEqual(strategy._target_capabilities, expected)

    def test_fitted_multilabel_classifier_rejected_before_state(self):
        self._test_fitted_multilabel_classifier_rejection(
            estimator_param="ensemble",
            fit_param="fit_ensemble",
            ensemble=True,
        )

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
            with self.subTest(init_labels=str(ensemble)):
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

    def test_tie_breaking_uses_random_state(self):
        probas = np.full((2, 4, 2), 0.5)
        utilities = [
            batch_bald(probas, batch_size=3, n_MC_samples=8, random_state=seed)
            for seed in (0, 2)
        ]
        self.assertFalse(
            np.array_equal(np.isnan(utilities[0]), np.isnan(utilities[1]))
        )
        np.testing.assert_array_equal(
            utilities[0],
            batch_bald(probas, batch_size=3, n_MC_samples=8, random_state=0),
        )

    def test_small_monte_carlo_budgets(self):
        probas = np.full((3, 5, 2), 0.5)
        for n_MC_samples in (1, 2, 3, 4, 5, 6, 7):
            with self.subTest(n_MC_samples=n_MC_samples):
                with np.errstate(all="raise"):
                    utilities = batch_bald(
                        probas,
                        batch_size=5,
                        n_MC_samples=n_MC_samples,
                        random_state=0,
                    )
                for step, row in enumerate(utilities):
                    self.assertEqual(np.isnan(row).sum(), step)
                    np.testing.assert_allclose(
                        row[~np.isnan(row)], 0, atol=1e-12
                    )

    def test_monte_carlo_sample_count_rounds_up_per_member(self):
        probas = np.full((2, 3, 2), 0.5)
        for requested, expected in ((1, 3), (2, 3), (3, 3), (4, 6), (7, 9)):
            with self.subTest(requested=requested):
                sampled = _SampledJointEntropy.sample(
                    probas, requested, np.random.RandomState(0)
                )
                self.assertEqual(
                    sampled.sampled_joint_probs_M_K.shape, (expected, 3)
                )

    def test_probability_inputs_are_preserved(self):
        values = np.array([[[1, 0], [0, 1], [1, 0]], [[0, 1], [0, 1], [0, 1]]])
        expected = batch_bald(
            values.astype(float), batch_size=3, n_MC_samples=8, random_state=0
        )
        for dtype, readonly in product(
            (np.float64, np.float32, np.int64), (False, True)
        ):
            with self.subTest(dtype=dtype, readonly=readonly):
                probas = values.astype(dtype)
                original = probas.copy()
                if readonly:
                    probas.setflags(write=False)
                actual = batch_bald(
                    probas, batch_size=3, n_MC_samples=8, random_state=0
                )
                np.testing.assert_array_equal(probas, original)
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_batch_bald(self):
        probas = np.random.RandomState(0).uniform(0.1, 0.9, (10, 100, 5))
        probas /= probas.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(
            _bald(probas), batch_bald(probas, batch_size=1)[0], rtol=1e-6
        )

    def test_batch_utilities_match_exact_enumeration(self):
        probas = np.random.RandomState(0).uniform(0.1, 0.9, (3, 4, 2))
        probas /= probas.sum(axis=-1, keepdims=True)
        utilities = batch_bald(
            probas, batch_size=4, n_MC_samples=8, random_state=0
        )
        for step, row in enumerate(utilities):
            selected = np.flatnonzero(np.isnan(row)).tolist()
            self.assertEqual(len(selected), step)
            for candidate in np.flatnonzero(~np.isnan(row)):
                joint = _joint_probabilities(probas[:, selected + [candidate]])
                expected = (
                    entropy(joint.mean(axis=0)) - entropy(joint, axis=1).mean()
                )
                self.assertAlmostEqual(row[candidate], expected)

    def test_sampled_joint_entropy_matches_exact_enumeration(self):
        probas = np.random.RandomState(0).uniform(0.1, 0.9, (3, 3, 2))
        probas /= probas.sum(axis=-1, keepdims=True)
        sampled = _SampledJointEntropy.sample(
            probas[:, :2].swapaxes(0, 1), 30000, np.random.RandomState(0)
        )
        actual = sampled.compute_batch(np.log(probas[:, 2:].swapaxes(0, 1)))
        expected = entropy(_joint_probabilities(probas).mean(axis=0))
        np.testing.assert_allclose(actual, [expected], rtol=0, atol=0.01)

    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
    ):
        for i, (test_val, err) in enumerate(test_cases):
            with self.subTest(msg="ID: {i}, Param", val=str(test_val)):
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


def _joint_probabilities(probas):
    """Compute joint label probabilities by enumerating all outcomes."""
    _, n_samples, n_classes = probas.shape
    return np.array(
        [
            np.prod(probas[:, np.arange(n_samples), labels], axis=1)
            for labels in product(range(n_classes), repeat=n_samples)
        ]
    ).T
