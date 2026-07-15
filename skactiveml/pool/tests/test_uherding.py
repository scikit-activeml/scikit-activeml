import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skactiveml.base import SkactivemlClassifier
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
from skactiveml.utils._validation import _canonicalize_multilabel_probas


class DummyMultilabelLogitClassifier(SkactivemlClassifier):
    fit_calls = 0

    def __init__(
        self,
        probas=None,
        logits=None,
        return_as_list=False,
        missing_label=MISSING_LABEL,
        target_type="multi-label",
    ):
        super().__init__(
            classes=[[0, 1], [0, 1]],
            missing_label=missing_label,
        )
        self.probas = probas
        self.logits = logits
        self.return_as_list = return_as_list
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        return frozenset(
            {("classification", "multi-label", "single-annotator")}
        )

    @classmethod
    def reset_fit_calls(cls):
        cls.fit_calls = 0

    def fit(self, X, y, sample_weight=None):
        type(self).fit_calls += 1
        target_spec = self._resolve_fitting_target_spec(y)
        self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            y_ensure_1d=False,
            multioutput_ensure_multilabel=True,
        )
        self.target_spec_ = target_spec
        self.is_fitted_ = True
        return self

    def _generate_logits(self, X):
        if self.logits is not None:
            logits = np.asarray(self.logits, dtype=float)
            if logits.ndim == 1:
                logits = np.tile(logits, (len(X), 1))
            return logits
        X = np.asarray(X, dtype=float)
        return np.column_stack([4 * X[:, 0] - 2, 2 - 4 * X[:, 1]])

    def predict_proba(
        self,
        X,
        return_logits=False,
        extra_outputs=None,
    ):
        logits = self._generate_logits(X)
        if self.probas is None:
            probas = expit(logits)
        else:
            probas = np.asarray(self.probas, dtype=float)
            if probas.ndim == 1:
                probas = np.tile(probas, (len(X), 1))
        if self.return_as_list:
            probas_out = [
                np.column_stack([1 - probas[:, j], probas[:, j]])
                for j in range(probas.shape[1])
            ]
        else:
            probas_out = probas

        if extra_outputs is not None:
            out = [probas_out]
            for name in extra_outputs:
                if name == "logits":
                    out.append(logits)
                else:
                    raise ValueError(f"Unsupported extra output `{name}`.")
            return tuple(out)
        if return_logits:
            return probas_out, logits
        return probas_out


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
        self.query_def_clf_multilabel = {
            "X": X,
            "y": np.vstack(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    *[
                        np.full(2, MISSING_LABEL, dtype=float)
                        for _ in range(len(X) - 2)
                    ],
                ]
            ),
            "clf": SklearnClassifier(
                estimator=MultiOutputClassifier(GaussianNB()),
                classes=[[0, 1], [0, 1]],
                missing_label=MISSING_LABEL,
                proba_format="array",
                random_state=0,
            ),
        }
        super().setUp(
            qs_class=UHerding,
            init_default_params={
                "predict_proba_dict": {"extra_outputs": ["logits", "emb"]},
                "random_state": 0,
            },
            query_default_params_clf=self.query_default_params_clf,
            init_default_params_multilabel={
                "predict_proba_dict": None,
                "random_state": 0,
            },
            query_default_params_clf_multilabel=self.query_def_clf_multilabel,
        )

    def test_target_contract(self):
        strategy = UHerding()

        self.assertEqual(strategy.target_type, "auto")
        self.assertEqual(
            strategy._target_capabilities,
            frozenset(
                {
                    (
                        "classification",
                        "single-output",
                        "single-annotator",
                    ),
                    ("classification", "multi-label", "single-annotator"),
                }
            ),
        )

    def test_fitted_target_spec_is_authoritative(self):
        X = self.query_default_params_clf_multilabel["X"]
        y = self.query_default_params_clf_multilabel["y"]
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            target_type="multi-label",
        ).fit(X, y)
        strategy = UHerding(target_type="single-output")

        with self.assertRaisesRegex(ValueError, "conflicts"):
            strategy.query(X, y, clf, fit_clf=False)

        self.assertFalse(hasattr(strategy, "n_features_in_"))

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
            (None, None),
            ({"extra_outputs": ["logits"]}, None),
            ({"extra_outputs": ["logits", "emb"]}, None),
            ({"test": True}, TypeError),
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
            ({"gamma": 2}, ValueError),
        ]
        self._test_param("init", "metric_dict", test_cases)
        test_cases = [
            ("gamma", TypeError),
            ({}, None),
            ({"gamma": 2}, None),
        ]
        self._test_param(
            "init",
            "metric_dict",
            test_cases,
            replace_init_params={"adaptive_sigma": False},
        )

    def test_init_param_multilabel_aggregation_fn(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.average, None), (np.max, None), ("bad", TypeError)]
        self._test_param("init", "multilabel_aggregation_fn", test_cases)

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
                None,
            ),
        ]
        self._test_param(
            "query",
            "clf",
            test_cases,
            replace_init_params={"predict_proba_dict": None},
        )

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

    def test_query_multilabel(self):
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        qs = UHerding(predict_proba_dict=None, random_state=0)
        query_indices, utilities = qs.query(
            **query_params, batch_size=2, return_utilities=True
        )
        self.assertEqual(query_indices.shape, (2,))
        self.assertEqual(utilities.shape, (2, len(query_params["X"])))
        self.assertTrue((utilities[~np.isnan(utilities)] >= 0).all())

        clf_list = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=MISSING_LABEL,
            proba_format="list",
            random_state=0,
        )
        query_params["clf"] = clf_list
        query_indices_list, utilities_list = qs.query(
            **query_params, batch_size=2, return_utilities=True
        )
        self.assertEqual(query_indices_list.shape, (2,))
        self.assertEqual(utilities_list.shape, (2, len(query_params["X"])))

    def test_query_multilabel_multiclass_list_probas_raises(self):
        X = np.linspace(0, 1, 12).reshape(6, 2)
        y = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1, 2], [0, 1]],
            missing_label=MISSING_LABEL,
            proba_format="list",
            random_state=0,
        )
        qs = UHerding(predict_proba_dict=None, random_state=0)
        self.assertRaises(ValueError, qs.query, X, y, clf=clf)

    def test_query_multilabel_fit_clf_false(self):
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        clf = query_params["clf"]
        clf.fit(query_params["X"], query_params["y"])
        query_params["clf"] = clf
        qs = UHerding(predict_proba_dict=None, random_state=0)
        query_indices, utilities = qs.query(
            **query_params,
            fit_clf=False,
            batch_size=2,
            return_utilities=True,
        )
        self.assertEqual(query_indices.shape, (2,))
        self.assertEqual(utilities.shape, (2, len(query_params["X"])))

    def test_query_multilabel_aggregation_changes_utilities(self):
        X = self.query_default_params_clf["X"]
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        query_params = {
            "X": X,
            "y": y,
            "clf": DummyMultilabelLogitClassifier(),
        }
        qs_mean = UHerding(
            predict_proba_dict=None,
            multilabel_aggregation_fn=np.average,
            random_state=0,
        )
        qs_max = UHerding(
            predict_proba_dict=None,
            multilabel_aggregation_fn=np.max,
            random_state=0,
        )
        _, utilities_mean = qs_mean.query(
            **query_params, batch_size=1, return_utilities=True
        )
        _, utilities_max = qs_max.query(
            **query_params, batch_size=1, return_utilities=True
        )
        self.assertGreater(
            np.nansum(np.abs(utilities_mean - utilities_max)), 0.0
        )

    def test_query_multilabel_with_logits(self):
        X = self.query_default_params_clf["X"]
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = DummyMultilabelLogitClassifier()
        query_indices, utilities = UHerding(
            predict_proba_dict={"extra_outputs": ["logits"]},
            random_state=0,
        ).query(X, y, clf=clf, batch_size=2, return_utilities=True)
        self.assertEqual(query_indices.shape, (2,))
        self.assertEqual(utilities.shape, (2, len(X)))
        self.assertTrue((utilities[~np.isnan(utilities)] >= 0).all())

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

    def test_query_multilabel_fixed_temperature_skips_calibration_refits(self):
        DummyMultilabelLogitClassifier.reset_fit_calls()
        X = self.query_default_params_clf["X"]
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = DummyMultilabelLogitClassifier()
        clf.fit(X, y)
        fit_calls_before = DummyMultilabelLogitClassifier.fit_calls

        qs = UHerding(temperatures=0.5, random_state=0)
        qs.query(X, y, clf=clf, fit_clf=False)
        self.assertEqual(
            DummyMultilabelLogitClassifier.fit_calls, fit_calls_before
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

    def test_query_zero_uncertainty_falls_back_to_pure_coverage(self):
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        )
        y = np.array([0, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(classes=self.classes, random_state=0)

        query_indices_uh, utilities_uh = UHerding(
            adaptive_sigma=False, random_state=0
        ).query(X, y, clf=clf, batch_size=2, return_utilities=True)
        query_indices_mh, utilities_mh = MaxHerding(random_state=0).query(
            X, y, batch_size=2, return_utilities=True
        )

        self.assertFalse(
            np.allclose(np.nan_to_num(utilities_uh, nan=0.0), 0.0)
        )
        np.testing.assert_array_equal(query_indices_uh, query_indices_mh)
        np.testing.assert_allclose(utilities_uh, utilities_mh)

    def test_query_multilabel_zero_uncertainty_falls_back_to_pure_coverage(
        self,
    ):
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        )
        y = np.array(
            [
                [0.0, 1.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = DummyMultilabelLogitClassifier(
            probas=np.array([1.0, 0.0]),
            logits=np.array([50.0, -50.0]),
        )

        query_indices_uh, utilities_uh = UHerding(
            adaptive_sigma=False, random_state=0
        ).query(X, y, clf=clf, batch_size=2, return_utilities=True)
        query_indices_mh, utilities_mh = MaxHerding(
            random_state=0, target_type="multi-label"
        ).query(X, y, batch_size=2, return_utilities=True)

        np.testing.assert_array_equal(query_indices_uh, query_indices_mh)
        np.testing.assert_allclose(utilities_uh, utilities_mh)

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

    def test_select_temperature_multilabel_missing_logits_and_split_fallback(
        self,
    ):
        qs = UHerding(predict_proba_dict=None, random_state=0)
        qs.missing_label_ = qs.missing_label
        qs.random_state_ = np.random.RandomState(0)
        query_params = deepcopy(self.query_default_params_clf_multilabel)

        tau = qs._select_temperature(
            query_params["X"],
            query_params["y"],
            query_params["clf"],
            temperatures=np.array([0.5, 1.0]),
            is_multilabel=True,
        )
        np.testing.assert_array_equal(tau, np.ones(2))

        y_small = np.array(
            [
                [0.0, 1.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = DummyMultilabelLogitClassifier()
        tau_small = qs._select_temperature(
            query_params["X"][:3],
            y_small,
            clf,
            temperatures=np.array([0.5, 1.0]),
            is_multilabel=True,
        )
        np.testing.assert_array_equal(tau_small, np.ones(2))

    def test_select_temperature_multilabel_degenerate_label(self):
        qs = UHerding(random_state=0)
        qs.missing_label_ = qs.missing_label
        qs.random_state_ = np.random.RandomState(0)
        X = self.query_default_params_clf["X"][:4]
        y = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        clf = DummyMultilabelLogitClassifier()

        with patch(
            "skactiveml.pool._uherding.train_test_split",
            return_value=(np.array([2, 3]), np.array([0, 1])),
        ):
            tau = qs._select_temperature(
                X,
                y,
                clf,
                temperatures=np.array([0.5, 1.0, 2.0]),
                is_multilabel=True,
            )

        self.assertEqual(tau.shape, (2,))
        self.assertEqual(tau[1], 1.0)
        self.assertIn(tau[0], [0.5, 1.0, 2.0])

    def test_select_temperature_multilabel_with_logits(self):
        qs = UHerding(
            predict_proba_dict={"extra_outputs": ["logits"]},
            random_state=0,
        )
        qs.missing_label_ = qs.missing_label
        qs.random_state_ = np.random.RandomState(0)
        X = self.query_default_params_clf["X"]
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        clf = DummyMultilabelLogitClassifier()

        with patch(
            "skactiveml.pool._uherding.train_test_split",
            return_value=(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
        ):
            tau = qs._select_temperature(
                X,
                y,
                clf,
                temperatures=np.array([0.5, 1.0, 2.0]),
                is_multilabel=True,
            )

        self.assertEqual(tau.shape, (2,))
        self.assertTrue(np.isin(tau, [0.5, 1.0, 2.0]).all())

    def test_predict_with_extras_multilabel_logits_only(self):
        qs = UHerding(
            predict_proba_dict={"extra_outputs": ["logits"]},
            predict_proba_parser=lambda out: (None, out[1], None),
            random_state=0,
        )
        clf = DummyMultilabelLogitClassifier()
        clf.fit(
            self.query_default_params_clf["X"],
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [MISSING_LABEL, MISSING_LABEL],
                    [MISSING_LABEL, MISSING_LABEL],
                    [MISSING_LABEL, MISSING_LABEL],
                    [MISSING_LABEL, MISSING_LABEL],
                ]
            ),
        )
        probas, logits, emb = qs._predict_with_extras(
            clf,
            self.query_default_params_clf["X"][:2],
            is_multilabel=True,
        )
        self.assertIsNone(emb)
        self.assertEqual(logits.shape, (2, 2))
        np.testing.assert_allclose(probas, expit(logits))

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

    def test_multilabel_helper_edge_cases(self):
        self.assertIsNone(
            _canonicalize_multilabel_probas(None, allow_none=True)
        )

        probas = _canonicalize_multilabel_probas(
            [
                np.array([[0.9, 0.1], [0.8, 0.2]]),
                np.array([[0.7, 0.3], [0.4, 0.6]]),
            ]
        )
        np.testing.assert_allclose(probas, np.array([[0.1, 0.3], [0.2, 0.6]]))

        self.assertRaises(
            ValueError,
            _canonicalize_multilabel_probas,
            [np.ones((2, 2, 1))],
        )
        self.assertRaises(
            ValueError,
            _canonicalize_multilabel_probas,
            np.ones(2),
        )

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
