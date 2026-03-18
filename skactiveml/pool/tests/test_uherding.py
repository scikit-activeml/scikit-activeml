import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import MaxHerding, UHerding
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import (
    ParzenWindowClassifierLogitsEmbedding,
    ParzenWindowClassifierLogitsEmbeddingTuple,
    ParzenWindowClassifierLogitsOnly,
    ParzenWindowClassifierWeirdTuple,
)
from skactiveml.utils import MISSING_LABEL


class TestUHerding(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.2],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.8, 0.8],
            ]
        )
        y = np.array([0, 1, 0, 1] + [MISSING_LABEL] * 4, dtype=float)
        self.query_default_params_clf = {
            "X": X,
            "y": y,
            "clf": ParzenWindowClassifierLogitsEmbedding(
                classes=self.classes, random_state=0
            ),
        }
        super().setUp(
            qs_class=UHerding,
            init_default_params={
                "predict_proba_dict": {"extra_outputs": ["logits", "emb"]},
                "random_state": 0,
            },
            query_default_params_clf=self.query_default_params_clf,
        )

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, ValueError),
            ("string", ValueError),
            ("least_confident", None),
            ("margin_sampling", None),
            ("entropy", None),
        ]
        self._test_param("init", "method", test_cases)

    def test_init_param_predict_proba_dict(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            (None, ValueError),
            ({"extra_outputs": ["logits"]}, None),
            ({"extra_outputs": ["logits", "emb"]}, None),
            ({"test": True}, (TypeError, ValueError)),
        ]
        self._test_param("init", "predict_proba_dict", test_cases)

    def test_init_param_predict_proba_parser(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            (None, None),
            (lambda out: out, None),
        ]
        self._test_param("init", "predict_proba_parser", test_cases)

    def test_init_param_temperatures(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (0.5, None),
            (0.0, ValueError),
            (-1.0, ValueError),
            (np.nan, ValueError),
            ([], ValueError),
            ([0.1, 1.0, 10.0], None),
            ([1.0], None),
            ([0.0, 1.0], ValueError),
            ([-1.0, 1.0], ValueError),
        ]
        self._test_param("init", "temperatures", test_cases)

    def test_init_param_validation_size(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("0.2", TypeError),
            (0.0, ValueError),
            (1.0, ValueError),
            (0.2, None),
            (1, None),
        ]
        self._test_param("init", "validation_size", test_cases)

    def test_init_param_n_ece_bins(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (0, ValueError),
            (1.5, TypeError),
            (10, None),
        ]
        self._test_param("init", "n_ece_bins", test_cases)

    def test_init_param_normalize_samples(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(False, None), (True, None), (1, TypeError)]
        self._test_param("init", "normalize_samples", test_cases)

    def test_init_param_adaptive_sigma(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(False, None), (True, None), (1, TypeError)]
        self._test_param("init", "adaptive_sigma", test_cases)

    def test_init_param_metric(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("rbf", None),
            ("linear", ValueError),
            (lambda x, y: ((x - y) ** 2).sum(), ValueError),
        ]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("gamma", TypeError),
            ({}, None),
            ({"gamma": 2}, None),
        ]
        self._test_param("init", "metric_dict", test_cases)

    def test_query_param_clf(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (SVC(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(SVC(probability=True)), None),
            (
                SklearnClassifier(LogisticRegression(), classes=self.classes),
                None,
            ),
            (
                ParzenWindowClassifier(classes=self.classes, random_state=0),
                (TypeError, ValueError),
            ),
        ]
        super().test_query_param_clf(test_cases=test_cases)

    def test_query(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.05, 0.15],
                [0.15, 0.95],
                [0.95, 0.15],
                [0.85, 0.95],
            ]
        )
        y = np.array([0, 1, 0, 1] + [MISSING_LABEL] * 4, dtype=float)

        qs_named = UHerding(
            predict_proba_dict={"extra_outputs": ["logits", "emb"]},
            random_state=0,
        )
        clf_named = ParzenWindowClassifierLogitsEmbedding(
            classes=self.classes, random_state=0
        )

        query_indices, utilities = qs_named.query(
            X, y, clf_named, batch_size=2, return_utilities=True
        )
        self.assertEqual(len(query_indices), 2)
        self.assertEqual(utilities.shape, (2, len(X)))
        self.assertTrue((utilities[~np.isnan(utilities)] >= 0).all())
        self.assertTrue(np.isnan(utilities[1, query_indices[0]]))

        candidates = np.arange(4, 8)
        _, utilities_idx = qs_named.query(
            X,
            y,
            clf_named,
            candidates=candidates,
            batch_size=2,
            return_utilities=True,
        )
        self.assertTrue(np.isnan(utilities_idx[:, :4]).all())
        self.assertFalse(np.isnan(utilities_idx[:, 4:8]).all())

        _, utilities_new = qs_named.query(
            X,
            y,
            clf_named,
            candidates=X[4:8],
            batch_size=2,
            return_utilities=True,
        )
        self.assertEqual(utilities_new.shape, (2, 4))

        qs_logits_only = UHerding(
            predict_proba_dict={"extra_outputs": ["logits"]},
            random_state=0,
        )
        _, utilities_logits_only = qs_logits_only.query(
            X, y, clf_named, batch_size=2, return_utilities=True
        )

        qs_tuple = UHerding(random_state=0)
        clf_tuple = ParzenWindowClassifierLogitsEmbeddingTuple(
            classes=self.classes, random_state=0
        )
        _, utilities_tuple = qs_tuple.query(
            X, y, clf_tuple, batch_size=2, return_utilities=True
        )

        np.testing.assert_allclose(utilities_logits_only, utilities_tuple)

        clf_weird = ParzenWindowClassifierWeirdTuple(
            classes=self.classes, random_state=0
        )
        qs_parser = UHerding(
            predict_proba_dict={"return_stuff": True},
            predict_proba_parser=lambda out: (out[0], out[2], out[1]),
            random_state=0,
        )
        _, utilities_parser = qs_parser.query(
            X, y, clf_weird, batch_size=2, return_utilities=True
        )
        np.testing.assert_allclose(utilities_tuple, utilities_parser)

        clf_logits_only = ParzenWindowClassifierLogitsOnly(
            classes=self.classes, random_state=0
        )
        qs_logits_parser = UHerding(
            predict_proba_dict={"return_logits": True},
            predict_proba_parser=lambda out: (None, out, None),
            random_state=0,
        )
        _, utilities_logits_parser = qs_logits_parser.query(
            X, y, clf_logits_only, batch_size=2, return_utilities=True
        )
        self.assertEqual(utilities_logits_parser.shape, (2, len(X)))
        self.assertTrue(
            (
                utilities_logits_parser[~np.isnan(utilities_logits_parser)]
                >= 0
            ).all()
        )

        clf_sklearn = SklearnClassifier(
            LogisticRegression(max_iter=1000),
            classes=self.classes,
            random_state=0,
            missing_label=MISSING_LABEL,
        )
        _, utilities_sklearn = UHerding(random_state=0).query(
            X, y, clf_sklearn, batch_size=2, return_utilities=True
        )
        self.assertEqual(utilities_sklearn.shape, (2, len(X)))
        self.assertTrue(
            (utilities_sklearn[~np.isnan(utilities_sklearn)] >= 0).all()
        )

        max_herding = MaxHerding(random_state=0, missing_label=MISSING_LABEL)
        _, utilities_max = max_herding.query(
            X, y, batch_size=1, return_utilities=True
        )
        self.assertGreater(
            np.nansum(np.abs(utilities[0] - utilities_max[0])), 0.0
        )

    def test_query_fit_clf_false_uses_temp_clones_only(self):
        ParzenWindowClassifierLogitsEmbedding.reset_fit_calls()
        clf = ParzenWindowClassifierLogitsEmbedding(
            classes=self.classes, random_state=0
        )
        clf.fit(
            self.query_default_params_clf["X"],
            self.query_default_params_clf["y"],
        )
        fit_calls_before = ParzenWindowClassifierLogitsEmbedding.fit_calls
        state_before = deepcopy(clf.__dict__)

        qs = UHerding(
            predict_proba_dict={"extra_outputs": ["logits", "emb"]},
            random_state=0,
        )
        qs.query(
            self.query_default_params_clf["X"],
            self.query_default_params_clf["y"],
            clf=clf,
            fit_clf=False,
        )

        self.assertGreater(
            ParzenWindowClassifierLogitsEmbedding.fit_calls, fit_calls_before
        )
        self._assert_object_dict_equal(state_before, clf.__dict__)

    def test_query_tau_fallback_when_split_infeasible(self):
        ParzenWindowClassifierLogitsEmbedding.reset_fit_calls()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        y = np.array([0, MISSING_LABEL, MISSING_LABEL], dtype=float)
        clf = ParzenWindowClassifierLogitsEmbedding(
            classes=self.classes, random_state=0
        )
        clf.fit(X, y)
        fit_calls_before = ParzenWindowClassifierLogitsEmbedding.fit_calls

        qs = UHerding(
            predict_proba_dict={"extra_outputs": ["logits", "emb"]},
            random_state=0,
        )
        query_indices, utilities = qs.query(
            X, y, clf=clf, fit_clf=False, return_utilities=True
        )

        self.assertEqual(len(query_indices), 1)
        self.assertEqual(utilities.shape, (1, len(X)))
        self.assertEqual(
            ParzenWindowClassifierLogitsEmbedding.fit_calls, fit_calls_before
        )

    def test_select_temperature_direct_return_for_fixed_temperature(self):
        qs = UHerding(random_state=0)
        qs.missing_label_ = qs.missing_label
        qs.random_state_ = np.random.RandomState(0)

        tau_scalar = qs._select_temperature(
            self.query_default_params_clf["X"],
            self.query_default_params_clf["y"],
            self.query_default_params_clf["clf"],
            temperatures=0.5,
        )
        self.assertEqual(tau_scalar, 0.5)

        tau_len_one = qs._select_temperature(
            self.query_default_params_clf["X"],
            self.query_default_params_clf["y"],
            self.query_default_params_clf["clf"],
            temperatures=np.array([0.25]),
        )
        self.assertEqual(tau_len_one, 0.25)

    def test_query_fixed_temperature_skips_calibration_refits(self):
        ParzenWindowClassifierLogitsEmbedding.reset_fit_calls()
        X = self.query_default_params_clf["X"]
        y = self.query_default_params_clf["y"]
        clf = ParzenWindowClassifierLogitsEmbedding(
            classes=self.classes, random_state=0
        )
        clf.fit(X, y)
        fit_calls_before = ParzenWindowClassifierLogitsEmbedding.fit_calls

        qs = UHerding(
            predict_proba_dict={"extra_outputs": ["logits", "emb"]},
            temperatures=0.5,
            random_state=0,
        )
        qs.query(X, y, clf=clf, fit_clf=False)
        self.assertEqual(
            ParzenWindowClassifierLogitsEmbedding.fit_calls, fit_calls_before
        )

    def test_query_no_logits_with_single_labeled_class(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        y = np.array([0, MISSING_LABEL, MISSING_LABEL], dtype=float)
        clf = ParzenWindowClassifier(classes=self.classes, random_state=0)

        query_indices, utilities = UHerding(random_state=0).query(
            X, y, clf=clf, return_utilities=True
        )
        self.assertEqual(len(query_indices), 1)
        self.assertEqual(utilities.shape, (1, len(X)))

    def test_select_temperature_second_split_failure(self):
        qs = UHerding(random_state=0)
        qs.missing_label_ = qs.missing_label
        qs.random_state_ = np.random.RandomState(0)
        X = self.query_default_params_clf["X"]
        y = self.query_default_params_clf["y"]
        clf = self.query_default_params_clf["clf"]

        with patch(
            "skactiveml.pool._uherding.train_test_split",
            side_effect=[ValueError("fail 1"), ValueError("fail 2")],
        ):
            tau = qs._select_temperature(
                X, y, clf, temperatures=np.array([0.5, 1.0])
            )
        self.assertEqual(tau, 1.0)

    def test_select_temperature_empty_split_and_fit_failure(self):
        qs = UHerding(random_state=0)
        qs.missing_label_ = qs.missing_label
        qs.random_state_ = np.random.RandomState(0)
        X = self.query_default_params_clf["X"]
        y = self.query_default_params_clf["y"]
        clf = self.query_default_params_clf["clf"]

        with patch(
            "skactiveml.pool._uherding.train_test_split",
            return_value=(np.array([], dtype=int), np.array([0], dtype=int)),
        ):
            tau = qs._select_temperature(
                X, y, clf, temperatures=np.array([0.5, 1.0])
            )
        self.assertEqual(tau, 1.0)

        class FailingClone:
            def fit(self, X, y, sample_weight=None):
                raise RuntimeError("fit failed")

        with patch(
            "skactiveml.pool._uherding.clone",
            return_value=FailingClone(),
        ):
            tau = qs._select_temperature(
                X, y, clf, temperatures=np.array([0.5, 1.0])
            )
        self.assertEqual(tau, 1.0)

    def test_predict_with_extras_type_error_without_kwargs_reraises(self):
        qs = UHerding(random_state=0)

        class TypeErrorClassifier:
            missing_label = MISSING_LABEL

            def predict_proba(self, X):
                raise TypeError("bad call")

        with self.assertRaises(TypeError):
            qs._predict_with_extras(TypeErrorClassifier(), np.zeros((2, 2)))

    def test_parse_predict_output_parser_edge_cases(self):
        qs_non_seq = UHerding(
            predict_proba_parser=lambda out: "not-a-sequence", random_state=0
        )
        with self.assertRaises(TypeError):
            qs_non_seq._parse_predict_output(np.ones((2, 2)))

        qs_len_two = UHerding(
            predict_proba_parser=lambda out: (out, out), random_state=0
        )
        probas, logits, emb = qs_len_two._parse_predict_output(np.ones((2, 2)))
        self.assertIsNone(emb)
        np.testing.assert_equal(probas, np.ones((2, 2)))
        np.testing.assert_equal(logits, np.ones((2, 2)))

        qs_bad_len = UHerding(
            predict_proba_parser=lambda out: (out, out, out, out),
            random_state=0,
        )
        with self.assertRaises(ValueError):
            qs_bad_len._parse_predict_output(np.ones((2, 2)))

    def test_parse_predict_output_tuple_edge_cases(self):
        qs = UHerding(random_state=0)
        with self.assertRaises(ValueError):
            qs._parse_predict_output(())
        with self.assertRaises(ValueError):
            qs._parse_predict_output(
                (
                    np.ones((2, 2)),
                    np.ones((2, 2)),
                    np.ones((2, 3)),
                    np.ones((2, 4)),
                )
            )

    def test_decision_function_logits_exception(self):
        class BadDecisionClassifier:
            def decision_function(self, X):
                raise RuntimeError("decision failed")

        logits = UHerding._decision_function_logits(
            BadDecisionClassifier(), np.zeros((2, 2))
        )
        self.assertIsNone(logits)

    @staticmethod
    def _assert_object_dict_equal(dict_a, dict_b):
        assert dict_a.keys() == dict_b.keys()
        for key in dict_a:
            val_a = dict_a[key]
            val_b = dict_b[key]
            if isinstance(val_a, np.ndarray):
                np.testing.assert_equal(val_a, val_b)
            elif isinstance(val_a, float) and np.isnan(val_a):
                assert isinstance(val_b, float) and np.isnan(val_b)
            elif isinstance(val_a, np.random.RandomState):
                assert isinstance(val_b, np.random.RandomState)
                state_a = val_a.get_state()
                state_b = val_b.get_state()
                assert state_a[0] == state_b[0]
                np.testing.assert_equal(state_a[1], state_b[1])
                assert state_a[2:] == state_b[2:]
            elif hasattr(val_a, "__dict__") and hasattr(val_b, "__dict__"):
                assert repr(val_a) == repr(val_b)
            else:
                assert val_a == val_b
