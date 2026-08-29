import unittest
import warnings
from copy import deepcopy
import inspect

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.regressor import SklearnRegressor
from skactiveml.pool import (
    SubSamplingWrapper,
    ParallelUtilityEstimationWrapper,
    DiscriminativeAL,
    QueryByCommittee,
    UncertaintySampling,
    RandomSampling,
)
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import assert_no_query_state
from skactiveml.utils import MISSING_LABEL, unlabeled_indices
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class DummyNonQueryStrategy:
    def query(self, **kwargs):
        pass


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
        # The wrapped strategy is estimator-backed, so that the multi-label
        # fixture carries a class vocabulary the wrapper must preserve.
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

    def test_target_contract_preserves_wrapped_strategy(self):
        wrapped = UncertaintySampling(method="entropy")
        wrapper = SubSamplingWrapper(query_strategy=wrapped)

        self.assertEqual(wrapper.target_type, "auto")
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

    def test_ambiguous_targets_fail_before_wrapper_state(self):
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

        assert_no_query_state(self, wrapper)

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

        conflicting = SubSamplingWrapper(
            query_strategy=UncertaintySampling(),
            max_candidates=4,
            target_type="single-output",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            conflicting.query(X, y_query, clf=clf, fit_clf=False)
        assert_no_query_state(self, conflicting)

    def test_fitted_estimator_vocabulary_fails_before_wrapper_state(self):
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
        y_query = y_fit.copy()
        y_query[0, 0] = 2.0
        wrapper = SubSamplingWrapper(
            query_strategy=UncertaintySampling(),
            max_candidates=4,
            random_state=0,
        )

        with self.assertRaisesRegex(ValueError, "outside `classes\\[0\\]`"):
            wrapper.query(X, y_query, clf=clf, fit_clf=False)

        assert_no_query_state(self, wrapper)

    def test_conflicting_fitted_estimators_fail_before_wrapper_state(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        clf_01 = SklearnClassifier(GaussianNB(), classes=[0, 1]).fit(
            X, [0, 1, 0, 1]
        )
        clf_02 = SklearnClassifier(GaussianNB(), classes=[0, 2]).fit(
            X, [0, 2, 0, 2]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=QueryByCommittee(), max_candidates=2
        )

        with self.assertRaisesRegex(ValueError, "conflicting target"):
            wrapper.query(
                X,
                np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
                ensemble=[clf_01, clf_02],
                fit_ensemble=False,
            )

        assert_no_query_state(self, wrapper)

    def test_explicit_estimator_target_types_conflict_before_query(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array(
            [
                [0, 1],
                [1, 0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=QueryByCommittee(),
            max_candidates=2,
            target_type="multi-label",
        )
        clf = SklearnClassifier(
            GaussianNB(), classes=[0, 1], target_type="single-output"
        )

        with self.assertRaisesRegex(
            ValueError, "target declaration conflicts"
        ):
            wrapper.query(X, y, ensemble=clf, fit_ensemble=False)

        assert_no_query_state(self, wrapper)

    def test_classifier_and_regressor_estimators_conflict_before_query(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        wrapper = SubSamplingWrapper(
            query_strategy=QueryByCommittee(), max_candidates=2
        )
        estimators = [
            SklearnClassifier(GaussianNB(), classes=[0, 1]),
            SklearnRegressor(RandomForestRegressor(random_state=0)),
        ]

        with self.assertRaisesRegex(
            ValueError, "conflicting classification and regression"
        ):
            wrapper.query(X, y, ensemble=estimators, fit_ensemble=False)

        assert_no_query_state(self, wrapper)

    def test_reordered_estimator_class_vocabularies_are_equivalent(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        # The same class vocabulary declared in a different order describes the
        # same targets, so wrapping must not change the selected candidates.
        estimators = [
            SklearnClassifier(GaussianNB(), classes=[1, 0]),
            SklearnClassifier(GaussianNB(), classes=[0, 1]),
        ]
        wrapper = SubSamplingWrapper(
            query_strategy=QueryByCommittee(random_state=0),
            max_candidates=4,
            random_state=0,
        )

        np.testing.assert_array_equal(
            wrapper.query(X, y, ensemble=estimators),
            QueryByCommittee(random_state=0).query(X, y, ensemble=estimators),
        )

    def test_wrapper_rejects_unsupported_explicit_target_type(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array(
            [
                [0, 1],
                [1, 0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=QueryByCommittee(),
            target_type="multi-label",
            max_candidates=2,
        )

        with self.assertRaisesRegex(ValueError, "does not support"):
            wrapper.query(X, y, ensemble=None)

        assert_no_query_state(self, wrapper)

    def test_resolved_target_type_reaches_auto_and_nested_strategies(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapped = RandomSampling(random_state=0)
        direct = SubSamplingWrapper(
            query_strategy=wrapped,
            max_candidates=4,
            target_type="multi-label",
            random_state=0,
        )

        direct_idx, direct_utilities = direct.query(
            X, y, return_utilities=True
        )

        self.assertIn(direct_idx[0], [2, 3, 4, 5])
        self.assertTrue(np.isnan(direct_utilities[0, :2]).all())
        self.assertEqual(wrapped.target_type, "auto")

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

    def test_target_type_declarations_conflict_before_query(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=RandomSampling(target_type="single-output"),
            max_candidates=4,
            target_type="multi-label",
        )

        with self.assertRaisesRegex(
            ValueError, "target declaration conflicts"
        ):
            wrapper.query(X, y)

        assert_no_query_state(self, wrapper)

    def test_wrapper_without_estimator_argument_resolves_declarations(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=RandomSampling(target_type="multi-label"),
            max_candidates=4,
            random_state=0,
        )

        target_type = wrapper._resolve_wrapped_target_type(y, {})

        self.assertEqual(target_type, "multi-label")
        self.assertEqual(wrapper.query(X, y).shape, (1,))

    def test_estimator_like_argument_is_no_target_authority(self):
        X = np.arange(24, dtype=float).reshape(12, 2)
        y_discriminator = np.array([0.0, 1.0] + [MISSING_LABEL] * 10)
        # The discriminator separates labeled from unlabeled samples such that
        # its target semantics do not describe `y`.
        discriminator = SklearnClassifier(GaussianNB(), classes=[0, 1]).fit(
            X, y_discriminator
        )
        y = np.array(
            [[0.0, 1.0], [1.0, 0.0]]
            + [[MISSING_LABEL, MISSING_LABEL] for _ in range(10)]
        )
        wrapper = SubSamplingWrapper(
            query_strategy=DiscriminativeAL(random_state=0),
            max_candidates=4,
            target_type="multi-label",
            random_state=0,
        )

        self.assertEqual(
            wrapper._collect_target_authorities(
                {"discriminator": discriminator}
            ),
            [],
        )
        query_indices = wrapper.query(X, y, discriminator=discriminator)

        self.assertNotIn(query_indices[0], [0, 1])

    def test_estimator_authorities_are_collected_deterministically(self):
        clf_0 = SklearnClassifier(GaussianNB(), classes=[0, 1])
        clf_1 = SklearnClassifier(GaussianNB(), classes=[0, 1])
        clf_2 = SklearnClassifier(GaussianNB(), classes=[0, 1])
        wrapper = SubSamplingWrapper(query_strategy=QueryByCommittee())

        authorities = wrapper._collect_target_authorities(
            {
                "ensemble": [clf_1, clf_2, clf_1],
                "clf": clf_0,
                "discriminator": SklearnClassifier(
                    GaussianNB(), classes=[0, 1]
                ),
                "fit_ensemble": True,
                "sample_weight": None,
            }
        )

        self.assertEqual(
            [id(a) for a in authorities],
            [id(clf_0), id(clf_1), id(clf_2)],
        )

    def test_authority_params_delegate_to_wrapped_strategy(self):
        wrapped = DiscriminativeAL()
        nested = SubSamplingWrapper(
            query_strategy=SubSamplingWrapper(query_strategy=wrapped)
        )

        self.assertEqual(
            nested._target_authority_params, wrapped._target_authority_params
        )
        self.assertNotIn("discriminator", wrapped._target_authority_params)
        self.assertEqual(SubSamplingWrapper()._target_authority_params, ())

    def test_conflicting_estimator_vocabularies_fail_before_query(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([0.0, 1.0, MISSING_LABEL, MISSING_LABEL])
        wrapper = SubSamplingWrapper(
            query_strategy=QueryByCommittee(random_state=0),
            max_candidates=2,
            random_state=0,
        )
        conflicting = [
            SklearnClassifier(GaussianNB(), classes=[0, 1]),
            SklearnClassifier(GaussianNB(), classes=[0, 2]),
        ]

        with self.assertRaisesRegex(ValueError, "class vocabularies"):
            wrapper.query(X, y, ensemble=conflicting, fit_ensemble=True)
        assert_no_query_state(self, wrapper)

        agreeing = [
            SklearnClassifier(GaussianNB(), classes=[0, 1]),
            SklearnClassifier(GaussianNB(), classes=[0, 1]),
        ]
        query_indices = wrapper.query(
            X, y, ensemble=agreeing, fit_ensemble=True
        )

        self.assertIn(query_indices[0], [2, 3])

    def test_cyclic_wrapper_chain_terminates(self):
        inner = SubSamplingWrapper(query_strategy=RandomSampling())
        outer = SubSamplingWrapper(
            query_strategy=inner, target_type="multi-label"
        )
        inner.query_strategy = outer

        declarations = outer._collect_target_declarations()

        self.assertEqual(
            declarations,
            [
                ("multi-label", "SubSamplingWrapper"),
                ("auto", "SubSamplingWrapper"),
            ],
        )

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
    # This wrapper parallelizes one-sample utility estimates and deliberately
    # rejects larger batches; its wrapped strategies retain their own batch
    # reproducibility coverage.
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
            # The wrapped strategy is estimator-backed, so that the multi-label
            # fixture carries a class vocabulary the wrapper must preserve.
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

        self.assertEqual(wrapper.target_type, "auto")
        self.assertEqual(
            wrapper._target_capabilities, wrapped._target_capabilities
        )

    def test_ambiguous_targets_fail_before_wrapper_state(self):
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

        assert_no_query_state(self, wrapper)

    def test_fitted_estimator_vocabulary_fails_before_wrapper_state(self):
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
        y_query = y_fit.copy()
        y_query[0, 0] = 2.0
        wrapper = ParallelUtilityEstimationWrapper(
            query_strategy=UncertaintySampling(),
            n_jobs=1,
            random_state=0,
        )

        with self.assertRaisesRegex(ValueError, "outside `classes\\[0\\]`"):
            wrapper.query(X, y_query, clf=clf, fit_clf=False)

        assert_no_query_state(self, wrapper)

    def test_resolved_target_type_reaches_auto_strategy(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )
        wrapped = RandomSampling(random_state=0)
        wrapper = ParallelUtilityEstimationWrapper(
            query_strategy=wrapped,
            n_jobs=1,
            target_type="multi-label",
            random_state=0,
        )

        query_idx, utilities = wrapper.query(X, y, return_utilities=True)

        self.assertIn(query_idx[0], [2, 3, 4, 5])
        self.assertTrue(np.isnan(utilities[0, :2]).all())
        self.assertEqual(wrapped.target_type, "auto")

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
        # With `n_jobs=-1`, the candidates are split across all available
        # CPUs. Fewer candidates than CPUs must not produce an empty chunk,
        # which would ask the wrapped strategy to select from an exhausted
        # candidate pool and contribute no utilities.
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

    def test_query_batch_variation(self):
        # The strategy does not support `batch_size > 1` (see documentation)
        pass

    def test_query_param_batch_size(self):
        super().test_query_param_batch_size(test_cases=[(2, ValueError)])
