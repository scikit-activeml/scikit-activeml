import unittest
from copy import deepcopy
from itertools import product

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.pool import UncertaintySampling, expected_average_precision
from skactiveml.pool._uncertainty_sampling import uncertainty_scores
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import assert_no_query_state


class TestUncertaintySampling(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "clf": ParzenWindowClassifier(
                random_state=0, classes=self.classes
            ),
            "y": np.array([0, 0, MISSING_LABEL, MISSING_LABEL]),
        }
        params_clf_multilabel = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "clf": SklearnClassifier(
                estimator=MultiOutputClassifier(GaussianNB()),
                classes=[[0, 1], [0, 1]],
                proba_format="array",
            ),
            "y": np.array(
                [[0.0, 1.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
            ),
        }
        super().setUp(
            qs_class=UncertaintySampling,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
            query_default_params_clf_multilabel=params_clf_multilabel,
        )

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1, TypeError), ("string", ValueError)]
        self._test_param("init", "method", test_cases)

    def test_init_param_cost_matrix(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.ones((2, 3)), ValueError),
            ("string", ValueError),
            (np.ones((3, 3)), ValueError),
        ]
        self._test_param("init", "cost_matrix", test_cases)
        self._test_param(
            "init",
            "cost_matrix",
            [(np.ones([2, 2]) - np.eye(2), ValueError)],
            replace_init_params={"method": "entropy"},
        )

    def test_init_param_multilabel_aggregation_fn(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.max, None), (np.average, None)]
        self._test_param("init", "multilabel_aggregation_fn", test_cases)

    def test_init_param_target_type(self):
        self._test_param(
            "init",
            "target_type",
            [
                ("auto", None),
                ("single-output", None),
                ("invalid", ValueError),
                (1, ValueError),
                ("multi-label", ValueError),
                ("multi-output", ValueError),
            ],
        )

    def test_capabilities_are_exact_and_configuration_dependent(self):
        standard = UncertaintySampling(method="entropy")
        average_precision = UncertaintySampling(
            method="expected_average_precision"
        )
        cost_sensitive = UncertaintySampling(
            method="least_confident", cost_matrix=[[0, 1], [1, 0]]
        )

        self.assertEqual(
            standard._target_capabilities,
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
        self.assertEqual(
            average_precision._target_capabilities,
            frozenset(
                {("classification", "single-output", "single-annotator")}
            ),
        )
        self.assertEqual(
            cost_sensitive._target_capabilities,
            frozenset(
                {("classification", "single-output", "single-annotator")}
            ),
        )
        self.assertNotIn(
            ("classification", "multi-label", "multi-annotator"),
            standard._target_capabilities,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (SVC(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(SVC(probability=True)), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_missing_label_mismatch_precedes_fit_flag_validation(self):
        query_params = deepcopy(self.query_default_params_clf)
        query_params["clf"] = ParzenWindowClassifier(
            classes=self.classes,
            missing_label=-1,
        )

        with self.assertRaisesRegex(ValueError, "must be equal"):
            UncertaintySampling().query(
                **query_params,
                fit_clf="invalid",
            )

    def test_query_param_sample_weight(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        X = self.query_default_params_clf["X"]
        test_cases += [
            ("string", ValueError),
            (X, ValueError),
            (np.empty((len(X) - 1)), ValueError),
        ]
        super().test_query_param_sample_weight(test_cases)

    def test_query_param_utility_weight(self):
        X = self.query_default_params_clf["X"]
        test_cases = [
            ("string", ValueError),
            (X, ValueError),
            (np.empty((len(X) - 1)), ValueError),
        ]
        self._test_param("query", "utility_weight", test_cases)
        self._test_param(
            "query",
            "utility_weight",
            [(np.ones(2), ValueError)],
            replace_query_params={"candidates": [2]},
        )
        self._test_param(
            "query",
            "utility_weight",
            [(np.ones(len(X) - 1), ValueError)],
            replace_query_params={"candidates": np.ones_like(X)},
        )

    def test_query(self):
        compare_list = []
        random_state = np.random.RandomState(42)
        clf = SklearnClassifier(
            estimator=GaussianProcessClassifier(),
            random_state=random_state,
            classes=self.classes,
        )
        candidates = random_state.rand(100, 10)
        X = random_state.rand(100, 10)
        y = random_state.randint(0, 2, (100,))

        # utility_weight
        qs = UncertaintySampling()
        utility_weight = np.arange(len(candidates))
        idx, utils_w = qs.query(
            X,
            y,
            clf,
            candidates=candidates,
            utility_weight=utility_weight,
            return_utilities=True,
        )
        idx, utils = qs.query(
            X, y, clf, candidates=candidates, return_utilities=True
        )
        np.testing.assert_array_equal(utils * utility_weight, utils_w)

        # query
        qs = UncertaintySampling(method="entropy")
        compare_list.append(qs.query(X, y, clf, candidates=candidates))

        qs = UncertaintySampling(method="margin_sampling")
        compare_list.append(qs.query(X, y, clf, candidates=candidates))

        qs = UncertaintySampling(method="least_confident")
        compare_list.append(qs.query(X, y, clf, candidates=candidates))

        for x in compare_list:
            self.assertEqual(compare_list[0], x)

        qs = UncertaintySampling(
            method="margin_sampling", cost_matrix=[[0, 1], [1, 0]]
        )
        qs.query(candidates=[[1]], clf=clf, X=[[1]], y=[MISSING_LABEL])

        qs = UncertaintySampling(
            method="least_confident", cost_matrix=[[0, 1], [1, 0]]
        )
        qs.query(candidates=[[1]], clf=clf, X=[[1]], y=[MISSING_LABEL])

        qs = UncertaintySampling(method="expected_average_precision")
        qs.query(candidates=[[1]], clf=clf, X=[[1]], y=[MISSING_LABEL])

        candidates = np.random.rand(10, 2)
        query_params = deepcopy(self.query_default_params_clf)
        query_params["candidates"] = candidates
        best_indices, utilities = qs.query(
            **query_params, return_utilities=True
        )
        self.assertEqual(utilities.shape, (1, len(candidates)))
        self.assertEqual(best_indices.shape, (1,))

    def test_query_multilabel_list_probas(self):
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        query_params["clf"] = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            proba_format="list",
        )

        query_idx, utilities = UncertaintySampling().query(
            **query_params, batch_size=2, return_utilities=True
        )
        self.assertEqual(query_idx.shape, (2,))
        self.assertEqual(utilities.shape, (2, len(query_params["X"])))
        self.assertTrue(np.isnan(utilities[:, :2]).all())

    def test_query_passes_sample_weight_to_classifier_fit(self):
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(classes=[0, 1])

        query_indices = UncertaintySampling().query(
            X=X,
            y=y,
            clf=clf,
            fit_clf=True,
            sample_weight=np.ones(len(y)),
            candidates=X[2:],
        )

        self.assertEqual(query_indices.shape, (1,))

    def test_query_reuses_fitted_multilabel_target_spec(self):
        X = np.arange(12, dtype=float).reshape(-1, 2)
        y_fit = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
        ).fit(X, y_fit)
        y_query = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        strategy = UncertaintySampling(target_type="auto", random_state=0)

        query_idx, utilities = strategy.query(
            X,
            y_query,
            clf,
            fit_clf=False,
            return_utilities=True,
        )

        self.assertIn(query_idx[0], [2, 3, 4, 5])
        self.assertTrue(np.isnan(utilities[0, :2]).all())
        self.assertFalse(hasattr(strategy, "target_spec_"))

    def test_query_rejects_values_outside_fitted_class_vocabularies(self):
        X = np.arange(12, dtype=float).reshape(-1, 2)
        y_fit = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            target_type="multi-label",
        ).fit(X, y_fit)
        y_query = y_fit.copy()
        y_query[0, 0] = 2
        strategy = UncertaintySampling()

        with self.assertRaisesRegex(ValueError, r"outside `classes\[0\]`"):
            strategy.query(X, y_query, clf, fit_clf=False)

        assert_no_query_state(self, strategy)

    def test_query_fits_explicit_multilabel_without_declared_classes(self):
        X = np.arange(12, dtype=float).reshape(-1, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
        )

        query_idx, utilities = UncertaintySampling(random_state=0).query(
            X, y, clf, return_utilities=True
        )

        self.assertIn(query_idx[0], [4, 5])
        self.assertTrue(np.isnan(utilities[0, :4]).all())

    def test_multilabel_capability_failure_precedes_acquisition_state(self):
        X = np.arange(8, dtype=float).reshape(-1, 2)
        y = np.array(
            [[0.0, 1.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            target_type="multi-label",
        ).fit(X, y)
        strategy = UncertaintySampling(method="expected_average_precision")

        with self.assertRaisesRegex(ValueError, "does not support"):
            strategy.query(X, y, clf, fit_clf=False)

        assert_no_query_state(self, strategy)

    def test_cost_sensitive_multilabel_methods_fail_before_acquisition_state(
        self,
    ):
        X = np.arange(8, dtype=float).reshape(-1, 2)
        y = np.array(
            [[0.0, 1.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            target_type="multi-label",
        ).fit(X, y)

        for method in ["least_confident", "margin_sampling", "entropy"]:
            with self.subTest(method=method):
                strategy = UncertaintySampling(
                    method=method, cost_matrix=[[0, 1], [1, 0]]
                )

                with self.assertRaisesRegex(ValueError, "does not support"):
                    strategy.query(X, y, clf, fit_clf=False)

                assert_no_query_state(self, strategy)

    def test_ambiguous_resolution_failure_precedes_acquisition_state(self):
        X = np.arange(8, dtype=float).reshape(-1, 2)
        y = np.array(
            [[0.0, 1.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
        )
        clf = SklearnClassifier(estimator=MultiOutputClassifier(GaussianNB()))
        strategy = UncertaintySampling()

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            strategy.query(X, y, clf, fit_clf=False)

        assert_no_query_state(self, strategy)

    def test_unfitted_classifier_declaration_conflict_precedes_state(self):
        X = np.arange(8, dtype=float).reshape(-1, 2)
        y = np.array(
            [[0.0, 1.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            target_type="single-output",
        )
        for fit_clf in [False, True]:
            with self.subTest(fit_clf=fit_clf):
                strategy = UncertaintySampling(target_type="multi-label")

                with self.assertRaisesRegex(ValueError, "explicit.*conflicts"):
                    strategy.query(X, y, clf, fit_clf=fit_clf)

                assert_no_query_state(self, strategy)

    def test_estimator_preflight_configuration_cross_product(self):
        X = np.arange(12, dtype=float).reshape(-1, 2)
        cost_matrix = [[0, 1], [1, 0]]
        target_cases = {
            "single-output": {
                "fit_y": np.array([0.0, 1.0, 0.0, 1.0, np.nan, np.nan]),
                "subset_y": np.array(
                    [0.0, 0.0, np.nan, np.nan, np.nan, np.nan]
                ),
                "outside_y": np.array(
                    [2.0, 0.0, np.nan, np.nan, np.nan, np.nan]
                ),
                "estimator": lambda: ParzenWindowClassifier(classes=[0, 1]),
            },
            "multi-label": {
                "fit_y": np.array(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 0.0],
                        [1.0, 1.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ]
                ),
                "subset_y": np.array(
                    [
                        [0.0, 1.0],
                        [0.0, 1.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ]
                ),
                "outside_y": np.array(
                    [
                        [2.0, 1.0],
                        [0.0, 1.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ]
                ),
                "estimator": lambda: SklearnClassifier(
                    estimator=MultiOutputClassifier(GaussianNB()),
                    classes=[[0, 1], [0, 1]],
                    target_type="multi-label",
                    proba_format="array",
                ),
            },
        }

        configurations = product(
            ["least_confident", "margin_sampling"],
            [False, True],
            target_cases,
            [False, True],
            ["subset", "outside"],
        )
        for (
            method,
            has_cost,
            target_type,
            is_fitted,
            vocabulary,
        ) in configurations:
            with self.subTest(
                method=method,
                has_cost=has_cost,
                target_type=target_type,
                is_fitted=is_fitted,
                vocabulary=vocabulary,
            ):
                case = target_cases[target_type]
                clf = case["estimator"]()
                if is_fitted:
                    clf.fit(X, case["fit_y"])
                y = case[f"{vocabulary}_y"]
                strategy = UncertaintySampling(
                    method=method,
                    cost_matrix=cost_matrix if has_cost else None,
                    target_type=target_type,
                    random_state=0,
                )

                unsupported = target_type == "multi-label" and has_cost
                if vocabulary == "outside" or unsupported:
                    with self.assertRaises(ValueError):
                        strategy.query(X, y, clf, fit_clf=not is_fitted)
                    assert_no_query_state(self, strategy)
                else:
                    query_idx = strategy.query(
                        X, y, clf, fit_clf=not is_fitted
                    )
                    self.assertIn(query_idx[0], range(2, len(X)))

    def test_default_single_output_classifier_query_remains_supported(self):
        X = np.arange(8, dtype=float).reshape(-1, 2)
        y = np.array([0.0, 1.0, np.nan, np.nan])
        clf = SklearnClassifier(estimator=GaussianNB()).fit(X, y)

        query_idx = UncertaintySampling(random_state=0).query(
            X, y, clf, fit_clf=False
        )

        self.assertEqual(clf.target_spec_.target_type, "single-output")
        self.assertIn(query_idx[0], [2, 3])

    def test_query_multilabel_multiclass_list_probas_raises(self):
        query_params = {
            "X": np.linspace(0, 1, 12).reshape(6, 2),
            "y": np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ]
            ),
            "clf": SklearnClassifier(
                estimator=MultiOutputClassifier(GaussianNB()),
                classes=[[0, 1, 2], [0, 1]],
                proba_format="list",
            ),
        }

        self.assertRaises(
            ValueError, UncertaintySampling().query, **query_params
        )


class TestExpectedAveragePrecision(unittest.TestCase):
    def setUp(self):
        self.classes = np.array([0, 1])
        self.probas = np.array([[0.4, 0.6], [0.3, 0.7]])
        self.scores_val = np.array([2.0, 2.0])

    def test_param_classes(self):
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=[],
            probas=self.probas,
        )
        self.assertRaises(
            TypeError,
            expected_average_precision,
            classes="string",
            probas=self.probas,
        )
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=[0],
            probas=self.probas,
        )
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=[0, 1, 2],
            probas=self.probas,
        )

    def test_param_probas(self):
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=self.classes,
            probas=[1],
        )
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=self.classes,
            probas=[[[1]]],
        )
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=self.classes,
            probas=[[0.7, 0.1, 0.2]],
        )
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=self.classes,
            probas=[[0.6, 0.2]],
        )
        self.assertRaises(
            ValueError,
            expected_average_precision,
            classes=self.classes,
            probas="string",
        )

    def test_expected_average_precision(self):
        expected_average_precision(classes=self.classes, probas=[[0.0, 1.0]])
        scores = expected_average_precision(
            classes=self.classes, probas=self.probas
        )
        self.assertTrue(scores.shape == (len(self.probas),))
        np.testing.assert_array_equal(scores, self.scores_val)


class TestUncertaintyScores(unittest.TestCase):
    def setUp(self):
        self.probas = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        self.multilabel_probas = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]])
        self.classes = np.array([0, 1, 2])
        self.cost_matrix = np.ones((3, 3))

    def test_param_probas(self):
        self.assertRaises(ValueError, uncertainty_scores, probas=[1])
        self.assertRaises(ValueError, uncertainty_scores, probas=[[[1]]])
        self.assertRaises(
            ValueError, uncertainty_scores, probas=[[0.6, 0.1, 0.2]]
        )
        self.assertRaises(ValueError, uncertainty_scores, probas="string")
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            probas=np.array([[1.1, 0.2], [0.4, 0.6]]),
            is_multilabel=True,
        )
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            probas=np.array([[-0.1, 0.2], [0.4, 0.6]]),
            is_multilabel=True,
        )
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            probas=[np.ones((2, 3)), np.ones((2, 2))],
            is_multilabel=True,
        )
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            probas=[np.ones((2, 2)), np.ones((3, 2))],
            is_multilabel=True,
        )

    def test_init_param_method(self):
        self.assertRaises(
            ValueError, uncertainty_scores, self.probas, method="String"
        )
        self.assertRaises(
            ValueError, uncertainty_scores, self.probas, method=1
        )
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            self.multilabel_probas,
            method="String",
            is_multilabel=True,
        )

    def test_param_cost_matrix(self):
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            self.probas,
            cost_matrix=np.ones((2, 3)),
        )
        self.assertRaises(
            ValueError, uncertainty_scores, self.probas, cost_matrix="string"
        )
        self.assertRaises(
            ValueError,
            uncertainty_scores,
            self.probas,
            cost_matrix=np.ones((2, 2)),
        )

    def test_uncertainty_scores(self):
        # least_confident
        val_scores = np.array([0.5, 0.3])
        scores = uncertainty_scores(self.probas, method="least_confident")
        np.testing.assert_allclose(val_scores, scores)
        # entropy
        val_scores = np.array([1.029653014, 0.8018185525])
        scores = uncertainty_scores(self.probas, method="entropy")
        np.testing.assert_allclose(val_scores, scores)
        # margin_sampling
        val_scores = np.array([0.8, 0.5])
        scores = uncertainty_scores(self.probas, method="margin_sampling")
        np.testing.assert_allclose(val_scores, scores)

        # multilabel methods
        val_scores = np.array([0.2333333333, 0.3333333333])
        scores = uncertainty_scores(
            self.multilabel_probas,
            method="least_confident",
            is_multilabel=True,
        )
        np.testing.assert_allclose(val_scores, scores)

        val_scores = np.array([0.5, 0.4])
        scores = uncertainty_scores(
            self.multilabel_probas,
            method="least_confident",
            is_multilabel=True,
            multilabel_aggregation_fn=np.max,
        )
        np.testing.assert_allclose(val_scores, scores)

        val_scores = np.array([0.4477710424, 0.6154752525])
        scores = uncertainty_scores(
            self.multilabel_probas, method="entropy", is_multilabel=True
        )
        np.testing.assert_allclose(val_scores, scores)

        val_scores = np.array([0.6931471804, 0.6730116670])
        scores = uncertainty_scores(
            self.multilabel_probas,
            method="entropy",
            is_multilabel=True,
            multilabel_aggregation_fn=np.max,
        )
        np.testing.assert_allclose(val_scores, scores)

        val_scores = np.array([0.4666666667, 0.6666666667])
        scores = uncertainty_scores(
            self.multilabel_probas,
            method="margin_sampling",
            is_multilabel=True,
        )
        np.testing.assert_allclose(val_scores, scores)

        val_scores = np.array([1.0, 0.8])
        scores = uncertainty_scores(
            self.multilabel_probas,
            method="margin_sampling",
            is_multilabel=True,
            multilabel_aggregation_fn=np.max,
        )
        np.testing.assert_allclose(val_scores, scores)

        multilabel_list = [
            np.column_stack(
                [
                    1 - self.multilabel_probas[:, 0],
                    self.multilabel_probas[:, 0],
                ]
            ),
            np.column_stack(
                [
                    1 - self.multilabel_probas[:, 1],
                    self.multilabel_probas[:, 1],
                ]
            ),
            np.column_stack(
                [
                    1 - self.multilabel_probas[:, 2],
                    self.multilabel_probas[:, 2],
                ]
            ),
        ]
        scores = uncertainty_scores(
            multilabel_list, method="entropy", is_multilabel=True
        )
        np.testing.assert_allclose(
            np.array([0.4477710424, 0.6154752525]), scores
        )
