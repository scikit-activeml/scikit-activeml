import unittest
import warnings
from copy import deepcopy
import inspect

import numpy as np
from sklearn import clone
from sklearn.datasets import load_breast_cancer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.regressor import SklearnRegressor, NICKernelRegressor
from skactiveml.pool import (
    SubSamplingWrapper,
    ParallelUtilityEstimationWrapper,
    QueryByCommittee,
    UncertaintySampling,
    RandomSampling,
    CoreSet,
    ProbCover,
    MonteCarloEER,
    ValueOfInformationEER,
    ExpectedModelOutputChange,
)
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import assert_attributes_unchanged
from skactiveml.utils import MISSING_LABEL, unlabeled_indices
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class DummyNonQueryStrategy:
    def query(self, **kwargs):
        pass


def _aligned_kwargs_setting():
    """Return samples, labels, and a classifier for weighted queries."""
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [-1.5, 2.0],
            [1.5, -2.0],
            [0.5, 0.5],
            [-0.5, -1.0],
        ]
    )
    y = np.full(len(X), MISSING_LABEL)
    y[0], y[1], y[2] = 0.0, 1.0, 0.0
    clf = ParzenWindowClassifier(classes=[0, 1])
    return X, y, clf


class TestSubSamplingWrapper(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        # Create dataset.
        X, y = load_breast_cancer(return_X_y=True)
        self.X = StandardScaler().fit_transform(X)
        y = y.astype(float)
        y[:50] = MISSING_LABEL
        y[350:] = MISSING_LABEL

        # Create test setting for classification.
        clf = SklearnClassifier(
            RandomForestClassifier(random_state=0),
            classes=[0, 1],
            missing_label=MISSING_LABEL,
            random_state=0,
        )
        query_default_params_clf = {
            "X": X,
            "y": y,
            "ensemble": clf,
            "fit_ensemble": True,
        }

        # Create test setting for regression.
        reg = SklearnRegressor(
            RandomForestRegressor(random_state=0),
            missing_label=MISSING_LABEL,
            random_state=0,
        )
        query_default_params_reg = {
            "X": X,
            "y": y,
            "ensemble": reg,
            "fit_ensemble": True,
        }

        # Setup initial parameters, where `QueryByCommittee` is used because
        # it can handle classification and regression models.
        init_default_params = {
            "query_strategy": QueryByCommittee(random_state=0),
            "max_candidates": 10,
            "exclude_non_subsample": False,
            "random_state": 0,
            "missing_label": MISSING_LABEL,
        }
        init_default_params_multilabel = {
            "query_strategy": UncertaintySampling(random_state=0),
            "max_candidates": 20,
        }
        params_clf_multilabel = {
            "X": X[:20],
            "y": np.vstack(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    *[
                        np.full(2, MISSING_LABEL, dtype=float)
                        for _ in range(18)
                    ],
                ]
            ),
            "clf": SklearnClassifier(
                MultiOutputClassifier(GaussianNB()),
                classes=[[0, 1], [0, 1]],
                missing_label=MISSING_LABEL,
                random_state=0,
            ),
        }

        super().setUp(
            qs_class=SubSamplingWrapper,
            init_default_params=init_default_params,
            init_default_params_multilabel=init_default_params_multilabel,
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
            query_default_params_clf_multilabel=params_clf_multilabel,
        )

    def test_refitting_respects_declared_and_inferred_vocabularies(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([0, 1, 2, MISSING_LABEL])
        strategies = [
            UncertaintySampling(),
            SubSamplingWrapper(UncertaintySampling(), max_candidates=4),
            ParallelUtilityEstimationWrapper(UncertaintySampling(), n_jobs=1),
            SingleAnnotatorWrapper(UncertaintySampling()),
            SingleAnnotatorWrapper(
                SubSamplingWrapper(UncertaintySampling(), max_candidates=4)
            ),
        ]
        for strategy in strategies:
            for classes in [None, [0, 1]]:
                for fit_clf in [None, True, False]:
                    with self.subTest(
                        strategy=type(strategy).__name__,
                        classes=classes,
                        fit_clf=fit_clf,
                    ):
                        qs = deepcopy(strategy)
                        clf = ParzenWindowClassifier(classes=classes).fit(
                            X[:2], [0, 1]
                        )
                        attributes_before = dict(clf.__dict__)
                        probabilities_before = clf.predict_proba(X)
                        kwargs = (
                            {} if fit_clf is None else {"fit_clf": fit_clf}
                        )
                        is_multiannotator = isinstance(
                            qs, SingleAnnotatorWrapper
                        )
                        query_y = (
                            np.column_stack([y, y]) if is_multiannotator else y
                        )

                        if classes is not None or fit_clf is False:
                            with self.assertRaisesRegex(
                                ValueError, "outside `classes`"
                            ):
                                qs.query(X, query_y, clf=clf, **kwargs)
                        else:
                            indices = qs.query(X, query_y, clf=clf, **kwargs)
                            samples = (
                                indices[:, 0] if is_multiannotator else indices
                            )
                            np.testing.assert_array_equal(samples, [3])

                        assert_attributes_unchanged(
                            self, clf, attributes_before
                        )
                        np.testing.assert_array_equal(
                            clf.predict_proba(X), probabilities_before
                        )

    def test_target_contract_preserves_wrapped_strategy(self):
        wrapped = UncertaintySampling(method="entropy")
        wrapper = SubSamplingWrapper(query_strategy=wrapped)

        self.assertEqual(
            wrapper._target_capabilities, wrapped._target_capabilities
        )
        restricted = UncertaintySampling(method="expected_average_precision")
        restricted_wrapper = SubSamplingWrapper(query_strategy=restricted)
        self.assertEqual(
            restricted_wrapper._target_capabilities,
            restricted._target_capabilities,
        )
        self.assertNotIn(
            ("classification", "multi-label", "single-annotator"),
            restricted_wrapper._target_capabilities,
        )

    def test_init_param_target_type(self):
        self.assertNotIn(
            "target_type", inspect.signature(self.qs_class).parameters
        )

    def test_query_multilabel_rejects_heterogeneous_vocabularies(self):
        init_params, query_params = self._multilabel_custom_vocabulary_params(
            self._multilabel_heterogeneous_vocabularies()
        )
        qs = self.qs_class(**init_params)

        with self.assertRaisesRegex(TypeError, "one label family"):
            qs.query(**query_params)

    def test_wrapped_strategy_reports_ambiguous_targets(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=RandomSampling(), max_candidates=2
        )

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            wrapper.query(X, y)

    def test_fitted_estimator_semantics_reach_wrapped_strategy(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y_fit = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = SklearnClassifier(
            MultiOutputClassifier(GaussianNB()), target_type="multi-label"
        ).fit(X, y_fit)
        y_query = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=UncertaintySampling(),
            max_candidates=4,
            random_state=0,
        )

        query_idx, utilities = wrapper.query(
            X,
            y_query,
            clf=clf,
            fit_clf=False,
            return_utilities=True,
        )

        self.assertIn(query_idx[0], [2, 3, 4, 5])
        self.assertEqual(utilities.shape, (1, len(X)))
        self.assertTrue(np.isnan(utilities[0, :2]).all())

    def test_query_does_not_reuse_wrapped_query_state(self):
        X = np.random.RandomState(0).uniform(size=(20, 2))
        y = np.full(len(X), MISSING_LABEL)
        y[:2] = [0, 1]
        wrapper = SubSamplingWrapper(
            ProbCover(n_classes=2, deltas=[0.35], random_state=0),
            max_candidates=0.5,
            exclude_non_subsample=True,
            random_state=0,
        )
        for cycle in range(3):
            query_idx = wrapper.query(X, y)
            np.testing.assert_array_equal(
                query_idx, clone(wrapper).query(X, y)
            )
            y[query_idx] = cycle % 2

    def test_multilabel_targets_reach_nested_strategies(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        nested = SubSamplingWrapper(
            query_strategy=SubSamplingWrapper(
                query_strategy=RandomSampling(
                    target_type="multi-label", random_state=0
                ),
                max_candidates=4,
                random_state=0,
            ),
            max_candidates=4,
            random_state=0,
        )
        nested_idx, nested_utilities = nested.query(
            X, y, return_utilities=True
        )

        self.assertIn(nested_idx[0], [2, 3, 4, 5])
        self.assertTrue(np.isnan(nested_utilities[0, :2]).all())

    def test_init_param_max_candidates(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (0, ValueError),
            (1.9, ValueError),
            ("", TypeError),
            (10, None),
            (0.9, None),
        ]
        self._test_param("init", "max_candidates", test_cases)

    def test_init_param_query_strategy(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (0, AttributeError),
            ("1.2", AttributeError),
            (QueryByCommittee(), None),
            (SingleAnnotatorWrapper(QueryByCommittee()), TypeError),
            (DummyNonQueryStrategy(), TypeError),
        ]
        self._test_param("init", "query_strategy", test_cases)

    def test_init_param_exclude_non_subsample(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (True, None),
            (False, None),
            (0, TypeError),
            (1, TypeError),
            ("1.2", TypeError),
            (DummyNonQueryStrategy(), TypeError),
        ]
        self._test_param("init", "exclude_non_subsample", test_cases)

    def test_init_param_embed_samples_func(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases

        def func_valid(x):
            return x

        def func_invalid(x, y):
            return x

        test_cases += [
            (None, None),
            (False, TypeError),
            (1, TypeError),
            ("1.2", TypeError),
            (func_valid, None),
            (func_invalid, TypeError),
        ]
        self._test_param("init", "embed_samples_func", test_cases)

    def test_query_param_sample_weight_aligned_to_samples(self):
        X, y, clf = _aligned_kwargs_setting()
        sample_weight = np.linspace(0.2, 2.0, len(X))
        candidate_indices = unlabeled_indices(y, MISSING_LABEL)
        for candidates in [None, candidate_indices, X[candidate_indices]]:
            results = []
            for exclude_non_subsample in [False, True]:
                qs = SubSamplingWrapper(
                    UncertaintySampling(random_state=0),
                    max_candidates=3,
                    exclude_non_subsample=exclude_non_subsample,
                    random_state=0,
                )
                results.append(
                    qs.query(
                        X=X,
                        y=y,
                        candidates=candidates,
                        clf=deepcopy(clf),
                        sample_weight=sample_weight,
                        return_utilities=True,
                    )
                )
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_allclose(results[0][1], results[1][1])

    def test_query_param_weights_aligned_to_candidates(self):
        X, y, clf = _aligned_kwargs_setting()
        candidates = X[unlabeled_indices(y, MISSING_LABEL)]
        weights = np.linspace(0.5, 3.0, len(candidates))
        for query_strategy, name in [
            (UncertaintySampling(random_state=0), "utility_weight"),
            (MonteCarloEER(random_state=0), "sample_weight_candidates"),
        ]:
            qs = SubSamplingWrapper(
                deepcopy(query_strategy),
                max_candidates=len(candidates),
                random_state=0,
            )
            indices, utilities = qs.query(
                X=X,
                y=y,
                candidates=candidates,
                clf=deepcopy(clf),
                return_utilities=True,
                **{name: weights},
            )
            expected_indices, expected_utilities = deepcopy(
                query_strategy
            ).query(
                X=X,
                y=y,
                candidates=candidates,
                clf=deepcopy(clf),
                return_utilities=True,
                **{name: weights},
            )
            np.testing.assert_array_equal(indices, expected_indices)
            np.testing.assert_allclose(utilities, expected_utilities)

    def test_query_param_utility_weight_aligned_to_samples(self):
        X, y, clf = _aligned_kwargs_setting()
        utility_weight = np.linspace(0.5, 3.0, len(X))
        candidate_indices = unlabeled_indices(y, MISSING_LABEL)
        for candidates in [None, candidate_indices]:
            results = []
            for exclude_non_subsample in [False, True]:
                qs = SubSamplingWrapper(
                    UncertaintySampling(random_state=0),
                    max_candidates=3,
                    exclude_non_subsample=exclude_non_subsample,
                    random_state=0,
                )
                results.append(
                    qs.query(
                        X=X,
                        y=y,
                        candidates=candidates,
                        clf=deepcopy(clf),
                        utility_weight=utility_weight,
                        return_utilities=True,
                    )
                )
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_allclose(results[0][1], results[1][1])

    def test_query_param_aligned_kwargs_leave_caller_arrays(self):
        X, y, clf = _aligned_kwargs_setting()
        candidates = X[unlabeled_indices(y, MISSING_LABEL)]
        sample_weight = np.linspace(0.2, 2.0, len(X))
        utility_weight = np.linspace(0.5, 3.0, len(candidates))
        expected_sample_weight = sample_weight.copy()
        expected_utility_weight = utility_weight.copy()
        qs = SubSamplingWrapper(
            UncertaintySampling(random_state=0),
            max_candidates=3,
            exclude_non_subsample=True,
            random_state=0,
        )
        qs.query(
            X=X,
            y=y,
            candidates=candidates,
            clf=deepcopy(clf),
            sample_weight=sample_weight,
            utility_weight=utility_weight,
        )
        np.testing.assert_array_equal(sample_weight, expected_sample_weight)
        np.testing.assert_array_equal(utility_weight, expected_utility_weight)

    def test_query_param_aligned_kwargs_of_wrong_length(self):
        X, y, clf = _aligned_kwargs_setting()
        qs = SubSamplingWrapper(
            UncertaintySampling(random_state=0),
            max_candidates=3,
            exclude_non_subsample=True,
            random_state=0,
        )
        self.assertRaisesRegex(
            ValueError,
            "sample_weight",
            qs.query,
            X=X,
            y=y,
            clf=deepcopy(clf),
            sample_weight=np.ones(len(X) - 2),
        )

    def test_query_param_aligned_kwargs_of_subset_length(self):
        X, y, clf = _aligned_kwargs_setting()
        n_labeled = int((~np.isnan(y)).sum())
        max_candidates = 3
        candidates = X[unlabeled_indices(y, MISSING_LABEL)]
        test_cases = [
            (
                UncertaintySampling,
                candidates,
                False,
                "utility_weight",
                max_candidates,
            ),
            (
                MonteCarloEER,
                candidates,
                False,
                "sample_weight_candidates",
                max_candidates,
            ),
            (
                UncertaintySampling,
                None,
                True,
                "sample_weight",
                n_labeled + max_candidates,
            ),
            (
                UncertaintySampling,
                None,
                True,
                "utility_weight",
                n_labeled + max_candidates,
            ),
            (
                UncertaintySampling,
                candidates,
                True,
                "sample_weight",
                n_labeled,
            ),
        ]
        for qs_class, candidates, exclude, name, n_values in test_cases:
            with self.subTest(
                name=name,
                candidates=np.ndim(candidates),
                exclude_non_subsample=exclude,
            ):
                qs = SubSamplingWrapper(
                    qs_class(random_state=0),
                    max_candidates=max_candidates,
                    exclude_non_subsample=exclude,
                    random_state=0,
                )
                self.assertRaisesRegex(
                    ValueError,
                    name,
                    qs.query,
                    X=X,
                    y=y,
                    candidates=candidates,
                    clf=deepcopy(clf),
                    **{name: np.ones(n_values)},
                )

    def test_query_param_candidates_embedded_like_samples(self):
        X = np.array(
            [
                [-2.0, 0.0],
                [-1.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
                [-1.5, 2.0],
                [1.5, -2.0],
                [0.5, 0.5],
            ]
        )
        y = np.full(len(X), MISSING_LABEL)
        y[0], y[1] = 0.0, 1.0
        candidate_indices = unlabeled_indices(y, MISSING_LABEL)
        embed_samples_funcs = [
            lambda x: x[:, :1],
            lambda x: x + 5.0,
            lambda x: x * np.array([1.0, 10.0]),
        ]
        strategies = [
            (
                UncertaintySampling(random_state=0),
                {"clf": ParzenWindowClassifier(classes=[0, 1])},
            ),
            (CoreSet(random_state=0), {}),
        ]
        for query_strategy, query_kwargs in strategies:
            for embed_samples_func in embed_samples_funcs:
                for candidates in [
                    None,
                    candidate_indices,
                    X[candidate_indices],
                ]:
                    for exclude_non_subsample in [False, True]:
                        init_params = {
                            "max_candidates": 3,
                            "exclude_non_subsample": exclude_non_subsample,
                            "random_state": 0,
                        }
                        qs = SubSamplingWrapper(
                            deepcopy(query_strategy),
                            embed_samples_func=embed_samples_func,
                            **init_params,
                        )
                        indices, utilities = qs.query(
                            X=X,
                            y=y,
                            candidates=candidates,
                            return_utilities=True,
                            **deepcopy(query_kwargs),
                        )
                        if candidates is None or candidates.ndim == 1:
                            expected_candidates = candidates
                        else:
                            expected_candidates = embed_samples_func(
                                candidates
                            )
                        qs_expected = SubSamplingWrapper(
                            deepcopy(query_strategy), **init_params
                        )
                        expected_indices, expected_utilities = (
                            qs_expected.query(
                                X=embed_samples_func(X),
                                y=y,
                                candidates=expected_candidates,
                                return_utilities=True,
                                **deepcopy(query_kwargs),
                            )
                        )
                        np.testing.assert_array_equal(
                            indices, expected_indices
                        )
                        np.testing.assert_allclose(
                            utilities, expected_utilities
                        )

    def test_query_param_X_eval_embedded_like_samples(self):
        X = np.array(
            [
                [-2.0, 0.0],
                [-1.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
                [-1.5, 2.0],
                [1.5, -2.0],
                [0.5, 0.5],
            ]
        )
        X_eval = np.array([[3.0, 3.0], [-3.0, -3.0], [0.5, 2.0]])
        y_class = np.full(len(X), MISSING_LABEL)
        y_class[0], y_class[1] = 0.0, 1.0
        y_reg = np.full(len(X), np.nan)
        y_reg[0], y_reg[1] = 0.3, 1.7
        embed_samples_funcs = [
            lambda x: x[:, :1],
            lambda x: x + 5.0,
            lambda x: x * np.array([1.0, 10.0]),
        ]
        strategies = [
            (
                MonteCarloEER(random_state=0),
                y_class,
                {"clf": ParzenWindowClassifier(classes=[0, 1])},
            ),
            (
                ExpectedModelOutputChange(random_state=0),
                y_reg,
                {"reg": NICKernelRegressor()},
            ),
        ]
        for query_strategy, y, query_kwargs in strategies:
            for embed_samples_func in embed_samples_funcs:
                for exclude_non_subsample in [False, True]:
                    init_params = {
                        "max_candidates": 3,
                        "exclude_non_subsample": exclude_non_subsample,
                        "random_state": 0,
                    }
                    qs = SubSamplingWrapper(
                        deepcopy(query_strategy),
                        embed_samples_func=embed_samples_func,
                        **init_params,
                    )
                    indices, utilities = qs.query(
                        X=X,
                        y=y,
                        X_eval=X_eval,
                        return_utilities=True,
                        **deepcopy(query_kwargs),
                    )
                    qs_expected = SubSamplingWrapper(
                        deepcopy(query_strategy), **init_params
                    )
                    expected_indices, expected_utilities = qs_expected.query(
                        X=embed_samples_func(X),
                        y=y,
                        X_eval=embed_samples_func(X_eval),
                        return_utilities=True,
                        **deepcopy(query_kwargs),
                    )
                    np.testing.assert_array_equal(indices, expected_indices)
                    np.testing.assert_allclose(utilities, expected_utilities)

    def test_query_param_X_eval_leaves_caller_array(self):
        X, y, clf = _aligned_kwargs_setting()
        X_eval = np.array([[3.0, 3.0], [-3.0, -3.0]])
        expected_X_eval = X_eval.copy()
        qs = SubSamplingWrapper(
            MonteCarloEER(random_state=0),
            max_candidates=3,
            embed_samples_func=lambda x: x * np.array([1.0, 10.0]),
            random_state=0,
        )
        qs.query(X=X, y=y, clf=deepcopy(clf), X_eval=X_eval)
        np.testing.assert_array_equal(X_eval, expected_X_eval)

    def test_query_param_embed_samples_func_preserving_dimensions(self):
        X = np.array(
            [
                [-2.0, 0.0],
                [-1.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
                [-1.5, 2.0],
            ]
        )
        y = np.full(len(X), MISSING_LABEL)
        y[0], y[1] = 0.0, 1.0
        embed_samples_funcs = [
            (lambda x: x[:, 0], "X"),
            (lambda x: x[:, 0] if len(x) < len(X) else x[:, :1], "candidates"),
        ]
        for embed_samples_func, name in embed_samples_funcs:
            qs = SubSamplingWrapper(
                UncertaintySampling(random_state=0),
                max_candidates=2,
                embed_samples_func=embed_samples_func,
                random_state=0,
            )
            self.assertRaisesRegex(
                ValueError,
                f"embedding for `{name}`",
                qs.query,
                X=X,
                y=y,
                candidates=X[2:],
                clf=ParzenWindowClassifier(classes=[0, 1]),
            )

    def test_query_param_embed_samples_func_preserving_samples(self):
        X = np.array(
            [
                [-2.0, 0.0],
                [-1.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
                [-1.5, 2.0],
            ]
        )
        y = np.full(len(X), MISSING_LABEL)
        y[0], y[1] = 0.0, 1.0
        for embed_samples_func in [
            lambda x: x[:-1],
            lambda x: np.vstack([x, x]),
        ]:
            qs = SubSamplingWrapper(
                UncertaintySampling(random_state=0),
                max_candidates=2,
                embed_samples_func=embed_samples_func,
                random_state=0,
            )
            for candidates in [None, X[2:]]:
                self.assertRaisesRegex(
                    ValueError,
                    "`embed_samples_func` returned",
                    qs.query,
                    X=X,
                    y=y,
                    candidates=candidates,
                    clf=ParzenWindowClassifier(classes=[0, 1]),
                )

    def test_query_param_query_kwargs(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(2, TypeError), (True, None), ("Hello", TypeError)]
        # Check is adjusted for ensemble since it is exemplary used
        # as parameters to the default query strategy of this test.
        self._test_param("query", "fit_ensemble", test_cases)

    def test_query(self):
        # check consistency with wrapped and non-wrapped query strategy
        qs = deepcopy(self.init_default_params["query_strategy"])
        qs_sub = self.qs_class(**deepcopy(self.init_default_params))

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            query_params = deepcopy(query_params)
            query_params["return_utilities"] = True
            query_params["batch_size"] = 1
            query_params["candidates"] = query_params["X"]
            (
                q,
                u,
            ) = qs.query(**query_params)
            u = u.ravel()
            q_sub, u_sub = qs_sub.query(**query_params)
            u_sub = u_sub.ravel()
            mask = ~np.isnan(u_sub) & ~np.isneginf(u_sub)
            np.testing.assert_array_equal(u[mask], u_sub[mask])
            query_params["return_utilities"] = False
            q_sub = qs_sub.query(**query_params)
            self.assertEqual(len(q_sub), 1)

        # check consistency of exclude_non_subsample with varying candidates
        # and batch_sizes for classification and regression
        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            init_params_base = deepcopy(self.init_default_params)
            init_params_base.pop("exclude_non_subsample")
            candidate_indices = unlabeled_indices(
                query_params["y"], init_params_base["missing_label"]
            )
            candidates_list = [
                None,
                candidate_indices,
                query_params["X"][candidate_indices],
            ]
            for batch_size in [1, 3]:
                for candidates in candidates_list:
                    if query_params is not None:
                        query_params_base = deepcopy(query_params)
                        query_params_base["return_utilities"] = True
                        query_params_base["candidates"] = candidates
                        query_params_base["batch_size"] = batch_size
                        qs_false = SubSamplingWrapper(
                            exclude_non_subsample=False, **init_params_base
                        )
                        qs_true = SubSamplingWrapper(
                            exclude_non_subsample=True, **init_params_base
                        )
                        query_indices_false, utilities_false = qs_false.query(
                            **query_params_base
                        )
                        query_indices_true, utilities_true = qs_true.query(
                            **query_params_base
                        )

                        np.testing.assert_array_equal(
                            query_indices_false, query_indices_true
                        )
                        np.testing.assert_array_equal(
                            utilities_false, utilities_true
                        )

        us = UncertaintySampling()
        for max_candidates in [0.2, len(self.X)]:
            qs_us = SubSamplingWrapper(
                us, max_candidates=max_candidates, exclude_non_subsample=True
            )
            qs_us.query(
                X=self.X,
                y=np.ones(len(self.X)),
                candidates=np.arange(len(self.X)),
                clf=ParzenWindowClassifier(),
            )
            sig_qs_us = inspect.signature(qs_us.query).parameters
            sig_us = inspect.signature(us.query).parameters
            self.assertEqual(sig_qs_us, sig_us)

    def test_query_batch_variation(self):
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                max_batch_size = qs.max_candidates
                batch_size = min(5, max_batch_size)
                self.assertTrue(batch_size > 1, msg="Too few unlabeled")

                query_params["batch_size"] = batch_size
                query_params["return_utilities"] = True
                query_ids, utils = qs.query(**query_params)

                self.assertEqual(len(query_ids), batch_size)
                self.assertEqual(len(utils), batch_size)
                self.assertEqual(len(utils[0]), len(query_params["X"]))
                self.assertEqual(
                    sum(~np.isneginf(utils[0]) & ~np.isnan(utils[0])),
                    qs.max_candidates,
                )

                query_params["batch_size"] = max_batch_size + 1
                query_params["return_utilities"] = False
                self.assertWarns(Warning, qs.query, **query_params)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    ids = qs.query(**query_params)
                    self.assertEqual(len(ids), max_batch_size)


class TestParallelUtilityEstimationWrapper(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    supports_multilabel_batch_variation = False
    reproducibility_batch_size = 1

    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = y.astype(float)
        y[:50] = MISSING_LABEL
        y[350:] = MISSING_LABEL
        clf = SklearnClassifier(
            RandomForestClassifier(random_state=0),
            classes=[0, 1],
            missing_label=MISSING_LABEL,
        )
        query_default_params_clf = {
            "X": X,
            "y": y,
            "ensemble": clf,
            "fit_ensemble": True,
        }

        # Create test setting for regression.
        reg = SklearnRegressor(
            RandomForestRegressor(random_state=0),
            missing_label=MISSING_LABEL,
            random_state=0,
        )
        query_default_params_reg = {
            "X": X,
            "y": y,
            "ensemble": reg,
            "fit_ensemble": True,
        }

        # Setup initial parameters, where `QueryByCommittee` is used because
        # it can handle classification and regression models.
        super().setUp(
            qs_class=ParallelUtilityEstimationWrapper,
            init_default_params={
                "query_strategy": QueryByCommittee(random_state=0),
                "n_jobs": 2,
            },
            init_default_params_multilabel={
                "query_strategy": UncertaintySampling(random_state=0),
                "n_jobs": 2,
            },
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
            query_default_params_clf_multilabel={
                "X": X[:20],
                "y": np.vstack(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                        *[
                            np.full(2, MISSING_LABEL, dtype=float)
                            for _ in range(18)
                        ],
                    ]
                ),
                "clf": SklearnClassifier(
                    MultiOutputClassifier(GaussianNB()),
                    classes=[[0, 1], [0, 1]],
                    missing_label=MISSING_LABEL,
                    random_state=0,
                ),
            },
        )

    def test_target_contract_preserves_wrapped_strategy(self):
        wrapped = RandomSampling(target_type="multi-label")
        wrapper = ParallelUtilityEstimationWrapper(query_strategy=wrapped)

        self.assertEqual(
            wrapper._target_capabilities, wrapped._target_capabilities
        )

    def test_init_param_target_type(self):
        self.assertNotIn(
            "target_type", inspect.signature(self.qs_class).parameters
        )

    def test_query_multilabel_rejects_heterogeneous_vocabularies(self):
        init_params, query_params = self._multilabel_custom_vocabulary_params(
            self._multilabel_heterogeneous_vocabularies()
        )
        qs = self.qs_class(**init_params)

        with self.assertRaisesRegex(TypeError, "one label family"):
            qs.query(**query_params)

    def test_wrapped_strategy_reports_ambiguous_targets(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapper = ParallelUtilityEstimationWrapper(
            query_strategy=RandomSampling(), n_jobs=1
        )

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            wrapper.query(X, y)

    def test_multilabel_targets_reach_wrapped_strategy(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapper = ParallelUtilityEstimationWrapper(
            query_strategy=RandomSampling(
                target_type="multi-label", random_state=0
            ),
            n_jobs=1,
            random_state=0,
        )

        query_idx, utilities = wrapper.query(X, y, return_utilities=True)

        self.assertIn(query_idx[0], [2, 3, 4, 5])
        self.assertTrue(np.isnan(utilities[0, :2]).all())

    def test_init_param_query_strategy(self):
        test_cases = [
            (QueryByCommittee(), None),
            (np.nan, AttributeError),
            ("state", AttributeError),
            (1.1, AttributeError),
            # Fails because test is using ensemble as input for the classifier
            (UncertaintySampling(), TypeError),
            (SingleAnnotatorWrapper(QueryByCommittee()), TypeError),
            (DummyNonQueryStrategy(), TypeError),
        ]
        self._test_param("init", "query_strategy", test_cases)

    def test_init_param_n_jobs(self):
        test_cases = [
            (2, None),
            (-1, None),
            (0, ValueError),
            ("multi", TypeError),
            ([0], TypeError),
        ]
        self._test_param("init", "n_jobs", test_cases)

    def test_init_param_parallel_dict(self):
        test_cases = [
            ({"backend": "threading"}, None),
            ({"backend": "loky", "batch_size": 2}, None),
            ({"backend": "loky", "batch_size": 2, "n_jobs": 1}, None),
            (0, TypeError),
            ("multi", TypeError),
            ([0], TypeError),
        ]
        self._test_param("init", "parallel_dict", test_cases)

    def test_query_param_query_kwargs(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(2, TypeError), (True, None), ("Hello", TypeError)]
        # Check is adjusted for clf since it is exemplary used
        # as parameters to the default query strategy of this test.
        self._test_param("query", "fit_ensemble", test_cases)

    def test_query(self):
        qs = deepcopy(self.init_default_params["query_strategy"])
        qs_sub = self.qs_class(**deepcopy(self.init_default_params))

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            query_params = deepcopy(query_params)
            query_params["return_utilities"] = True
            query_params["batch_size"] = 1
            query_params["candidates"] = query_params["X"]
            (
                q,
                u,
            ) = qs.query(**query_params)
            u = u.ravel()
            q_sub, u_sub = qs_sub.query(**query_params)
            u_sub = u_sub.ravel()
            mask = ~np.isnan(u_sub)
            np.testing.assert_array_equal(u[mask], u_sub[mask])
            query_params["return_utilities"] = False
            q_sub = qs_sub.query(**query_params)
            self.assertEqual(len(q_sub), 1)

    def test_query_fewer_candidates_than_jobs(self):
        query_params = deepcopy(self.query_default_params_clf)
        candidates = unlabeled_indices(
            query_params["y"], self.init_default_params["missing_label"]
        )[:2]
        query_params["candidates"] = candidates
        query_params["return_utilities"] = True

        init_params = deepcopy(self.init_default_params)
        init_params["n_jobs"] = -1
        query_indices, utilities = self.qs_class(**init_params).query(
            **query_params
        )

        self.assertEqual(query_indices.shape, (1,))
        self.assertIn(query_indices[0], candidates)
        self.assertEqual(utilities.shape, (1, len(query_params["X"])))
        self.assertEqual(int((~np.isnan(utilities[0])).sum()), len(candidates))

    def test_query_param_weights_match_direct_query(self):
        X, y, clf = _aligned_kwargs_setting()
        candidate_indices = unlabeled_indices(y, MISSING_LABEL)
        candidate_samples = X[candidate_indices]
        test_cases = [
            (UncertaintySampling, None, "utility_weight", len(X)),
            (UncertaintySampling, candidate_indices, "utility_weight", len(X)),
            (
                UncertaintySampling,
                candidate_samples,
                "utility_weight",
                len(candidate_samples),
            ),
            (
                MonteCarloEER,
                candidate_samples,
                "sample_weight_candidates",
                len(candidate_samples),
            ),
        ]
        for qs_class, candidates, name, n_weights in test_cases:
            query_params = {
                "X": X,
                "y": y,
                "candidates": candidates,
                "return_utilities": True,
                name: np.linspace(0.5, 3.0, n_weights),
            }
            expected_indices, expected_utilities = qs_class(
                random_state=0
            ).query(clf=deepcopy(clf), **query_params)
            for n_jobs in [1, 2]:
                with self.subTest(
                    qs_class=qs_class.__name__,
                    candidates=np.ndim(candidates),
                    n_jobs=n_jobs,
                ):
                    qs = ParallelUtilityEstimationWrapper(
                        qs_class(random_state=0),
                        n_jobs=n_jobs,
                        random_state=0,
                    )
                    indices, utilities = qs.query(
                        clf=deepcopy(clf), **query_params
                    )
                    np.testing.assert_array_equal(indices, expected_indices)
                    np.testing.assert_allclose(utilities, expected_utilities)

    def test_query_index_candidates_match_direct_query(self):
        X, y, clf = _aligned_kwargs_setting()
        test_cases = [
            (ValueOfInformationEER, None),
            (ValueOfInformationEER, [0, 3, 5, 7]),
            (MonteCarloEER, [0, 3, 5, 7]),
        ]
        for qs_class, candidates in test_cases:
            query_params = {
                "X": X,
                "y": y,
                "candidates": candidates,
                "return_utilities": True,
            }
            expected_indices, expected_utilities = qs_class(
                random_state=0
            ).query(clf=deepcopy(clf), **query_params)
            for n_jobs in [1, 2]:
                with self.subTest(
                    qs_class=qs_class.__name__,
                    candidates=candidates,
                    n_jobs=n_jobs,
                ):
                    qs = ParallelUtilityEstimationWrapper(
                        qs_class(random_state=0),
                        n_jobs=n_jobs,
                        random_state=0,
                    )
                    indices, utilities = qs.query(
                        clf=deepcopy(clf), **query_params
                    )
                    np.testing.assert_array_equal(indices, expected_indices)
                    np.testing.assert_allclose(utilities, expected_utilities)

    def test_query_param_weights_of_wrong_length(self):
        X, y, clf = _aligned_kwargs_setting()
        candidates = X[unlabeled_indices(y, MISSING_LABEL)]
        for n_weights in [len(X), len(candidates) // 2]:
            for qs_class, name in [
                (UncertaintySampling, "utility_weight"),
                (MonteCarloEER, "sample_weight_candidates"),
            ]:
                with self.subTest(name=name, n_weights=n_weights):
                    qs = ParallelUtilityEstimationWrapper(
                        qs_class(random_state=0), n_jobs=2
                    )
                    self.assertRaisesRegex(
                        ValueError,
                        name,
                        qs.query,
                        X=X,
                        y=y,
                        candidates=candidates,
                        clf=deepcopy(clf),
                        **{name: np.ones(n_weights)},
                    )

    def test_query_batch_variation(self):
        # The strategy does not support `batch_size > 1` (see documentation)
        pass

    def test_query_param_batch_size(self):
        super().test_query_param_batch_size(test_cases=[(2, ValueError)])
