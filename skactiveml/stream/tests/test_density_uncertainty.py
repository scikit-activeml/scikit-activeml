import unittest
from copy import deepcopy

import numpy as np
from skactiveml.base import BudgetManager
from sklearn.metrics.pairwise import manhattan_distances, pairwise_distances
from skactiveml.utils import MISSING_LABEL
from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.stream.budgetmanager import (
    DensityBasedSplitBudgetManager,
    FixedUncertaintyBudgetManager,
)
from skactiveml.stream import (
    StreamDensityBasedAL,
    CognitiveDualQueryStrategy,
    CognitiveDualQueryStrategyRan,
    CognitiveDualQueryStrategyRanVarUn,
    CognitiveDualQueryStrategyVarUn,
    CognitiveDualQueryStrategyFixUn,
)

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorStreamQueryStrategy,
)
from skactiveml.tests.utils import assert_state_unchanged


class RecordingBudgetManager(BudgetManager):
    """
    A utility-dependent manager exercising the public extension
    interface.
    """

    def __init__(self):
        super().__init__(budget=0.5)
        self.history_ = []

    def query_by_utility(self, utilities):
        return (
            [0]
            if np.isfinite(utilities[0]) and len(self.history_) % 2 == 0
            else []
        )

    def update(self, candidates, queried_indices, utilities, marker=None):
        if len(candidates) != len(utilities):
            raise ValueError("Utilities must align with observed candidates.")
        for i, candidate in enumerate(candidates):
            self.history_.append(
                (
                    np.asarray(candidate).copy(),
                    i in queried_indices,
                    utilities[i],
                    marker,
                )
            )
        return self


class FailingBudgetManager(RecordingBudgetManager):
    def update(self, candidates, queried_indices, utilities, marker=None):
        super().update(candidates, queried_indices, utilities, marker)
        if len(self.history_) >= 2:
            raise RuntimeError("Cannot advance this manager.")
        return self


def _empty_classifier():
    return ParzenWindowClassifier(classes=[0, 1]).fit([[0.0]], [np.nan])


def _assert_acquisition_state_equal(test_case, actual, expected):
    assert_state_unchanged(
        test_case,
        actual.budget_manager_.__dict__,
        expected.budget_manager_.__dict__,
        name="budget_manager",
    )
    for name in (
        "window_",
        "min_dist_",
        "cognition_window_",
        "theta_",
        "s_",
        "t_x_",
        "f_",
        "t_",
        "random_state_",
    ):
        if hasattr(actual, name):
            assert_state_unchanged(
                test_case,
                getattr(actual, name),
                getattr(expected, name),
                name=name,
            )


def _assert_batch_accounts_for_earlier_acquisitions(test_case, strategy_type):
    candidates = (1 - 2.0 ** (-np.arange(20)))[:, None]
    kwargs = (
        {"force_full_budget": True}
        if strategy_type is CognitiveDualQueryStrategy
        else {}
    )
    strategy = strategy_type(
        budget_manager=DensityBasedSplitBudgetManager(
            budget=0.1, delta=0.001, random_state=0
        ),
        random_state=0,
        **kwargs,
    )
    queried = strategy.query(candidates, _empty_classifier())
    test_case.assertEqual(queried, [1, 10])
    strategy.update(candidates, queried)
    test_case.assertEqual(strategy.budget_manager_.t_, 20)
    test_case.assertEqual(strategy.budget_manager_.u_, 2)


def _assert_invalid_indices_do_not_advance_state(test_case, strategy_type):
    strategy = strategy_type(random_state=0)
    strategy.update([[0.0]], [])
    before = deepcopy(strategy)
    for indices in ([-1], [2], [0.5], [[0]], [True]):
        with test_case.subTest(indices=indices):
            with test_case.assertRaises(IndexError):
                strategy.update([[0.5], [0.75]], indices)
            _assert_acquisition_state_equal(test_case, strategy, before)


def _assert_query_isolates_rng_and_preserves_state(test_case, strategy_type):
    candidates = (1 - 2.0 ** (-np.arange(20)))[:, None]
    classifier = _empty_classifier()
    kwargs = (
        {"force_full_budget": True}
        if strategy_type is CognitiveDualQueryStrategy
        else {}
    )
    original_global_state = np.random.get_state()
    try:
        results = []
        for boundaries in (list(range(1, 21)), [20], [4, 9, 20]):
            np.random.seed(17)
            strategy = strategy_type(
                budget_manager=DensityBasedSplitBudgetManager(
                    budget=0.3, delta=0.7, random_state=None
                ),
                random_state=0,
                **kwargs,
            )
            selected = []
            start = 0
            for stop in boundaries:
                batch = candidates[start:stop]
                global_before = np.random.get_state()
                queried, utilities = strategy.query(
                    batch, classifier, return_utilities=True
                )
                assert_state_unchanged(
                    test_case,
                    np.random.get_state(),
                    global_before,
                    name="global_random_state",
                )
                before = deepcopy(strategy)
                queried_again, utilities_again = strategy.query(
                    batch, classifier, return_utilities=True
                )
                test_case.assertEqual(queried_again, queried)
                np.testing.assert_array_equal(utilities_again, utilities)
                _assert_acquisition_state_equal(test_case, strategy, before)
                strategy.update(batch, queried)
                selected.extend(start + i for i in queried)
                start = stop
            results.append(
                (selected, deepcopy(strategy), np.random.get_state())
            )
        for result in results[1:]:
            test_case.assertEqual(result[0], results[0][0])
            _assert_acquisition_state_equal(
                test_case, result[1], results[0][1]
            )
            assert_state_unchanged(
                test_case,
                result[2],
                results[0][2],
                name="global_random_state",
            )
    finally:
        np.random.set_state(original_global_state)


def _assert_query_failures_restore_state(test_case, strategy_type):
    def failing_distance(X, Y):
        if len(X) >= 2:
            raise RuntimeError("Cannot measure this distance.")
        return pairwise_distances(X, Y)

    classifier = _empty_classifier()
    for failure in ("distance", "manager"):
        with test_case.subTest(failure=failure):
            kwargs = (
                {"force_full_budget": True}
                if strategy_type is CognitiveDualQueryStrategy
                else {}
            )
            if failure == "distance":
                kwargs["dist_func"] = failing_distance
                kwargs["budget_manager"] = DensityBasedSplitBudgetManager(
                    random_state=None
                )
            else:
                kwargs["budget_manager"] = FailingBudgetManager()
            strategy = strategy_type(random_state=0, **kwargs)
            strategy.query([[0.0]], classifier)
            before = deepcopy(strategy)
            global_before = np.random.get_state()
            with test_case.assertRaisesRegex(RuntimeError, "Cannot"):
                strategy.query([[0.0], [0.5], [0.75]], classifier)
            _assert_acquisition_state_equal(test_case, strategy, before)
            assert_state_unchanged(
                test_case,
                np.random.get_state(),
                global_before,
                name="global_random_state",
            )


def _assert_partition_invariance(test_case, strategy_type):
    candidates = np.concatenate(
        [
            np.repeat([0.0, 0.5, 0.75, 0.875], 2),
            np.linspace(0.9, 0.999, 24),
        ]
    )[:, None]
    classifier = _empty_classifier()
    modes = [True] if strategy_type is StreamDensityBasedAL else [False, True]
    original_global_state = np.random.get_state()
    initial_global_state = np.random.RandomState(17).get_state()
    try:
        for full in modes:
            for seed_type in ("integer", "instance", "global"):
                with test_case.subTest(full=full, seed=seed_type):
                    results = []
                    for boundaries in (
                        list(range(1, 33)),
                        [32],
                        [3, 8, 19, 32],
                    ):
                        np.random.set_state(initial_global_state)
                        seed = (
                            0
                            if seed_type == "integer"
                            else (
                                np.random.RandomState(0)
                                if seed_type == "instance"
                                else None
                            )
                        )
                        kwargs = {"random_state": seed, "budget": 0.3}
                        if strategy_type is not StreamDensityBasedAL:
                            kwargs["force_full_budget"] = full
                        if strategy_type is CognitiveDualQueryStrategyFixUn:
                            kwargs["classes"] = [0, 1]
                        strategy = strategy_type(**kwargs)
                        selected, utilities = [], []
                        start = 0
                        for stop in boundaries:
                            batch = candidates[start:stop]
                            queried, batch_utilities = strategy.query(
                                batch,
                                classifier,
                                return_utilities=True,
                            )
                            strategy.update(batch, queried)
                            selected.extend(start + i for i in queried)
                            utilities.extend(batch_utilities)
                            start = stop
                        results.append(
                            (selected, utilities, deepcopy(strategy))
                        )
                    for result in results[1:]:
                        test_case.assertEqual(result[0], results[0][0])
                        np.testing.assert_array_equal(result[1], results[0][1])
                        _assert_acquisition_state_equal(
                            test_case, result[2], results[0][2]
                        )
    finally:
        np.random.set_state(original_global_state)


def _assert_dist_func_dict_is_forwarded(test_case, strategy_type):
    candidates = np.random.RandomState(0).uniform(size=(30, 2))
    classifier = ParzenWindowClassifier(classes=[0, 1]).fit(
        [[0.0, 0.0]], [np.nan]
    )
    kwargs = {"random_state": 0, "budget": 0.3}
    if strategy_type is not StreamDensityBasedAL:
        kwargs["force_full_budget"] = True
    if strategy_type is CognitiveDualQueryStrategyFixUn:
        kwargs["classes"] = [0, 1]

    def distance(X, Y, *, metric):
        return pairwise_distances(X, Y, metric=metric)

    results = []
    for dist_params in (
        {"dist_func": distance, "dist_func_dict": {"metric": "manhattan"}},
        {"dist_func": manhattan_distances},
        {},
        {"dist_func": distance, "dist_func_dict": {"metric": "euclidean"}},
    ):
        strategy = strategy_type(**kwargs, **dist_params)
        selected = []
        for start in (0, 15):
            batch = candidates[start : start + 15]
            queried = strategy.query(batch, classifier)
            strategy.update(batch, queried)
            selected.extend(start + i for i in queried)
        results.append((selected, strategy))
    test_case.assertEqual(results[0][0], results[1][0])
    _assert_acquisition_state_equal(test_case, results[0][1], results[1][1])
    test_case.assertFalse(
        np.allclose(results[0][1].min_dist_, results[2][1].min_dist_)
    )
    test_case.assertEqual(results[2][0], results[3][0])
    _assert_acquisition_state_equal(test_case, results[2][1], results[3][1])


class TemplateCognitiveDualQueryStrategy(
    TemplateSingleAnnotatorStreamQueryStrategy
):
    def test_partition_invariance(self):
        _assert_partition_invariance(self, self.qs_class)

    def test_dist_func_dict_is_forwarded(self):
        _assert_dist_func_dict_is_forwarded(self, self.qs_class)

    def test_cognition_window_holds_at_most_its_size(self):
        candidates = np.random.RandomState(0).uniform(size=(12, 1))
        kwargs = {"force_full_budget": True, "random_state": 0}
        if self.qs_class is CognitiveDualQueryStrategyFixUn:
            kwargs["classes"] = [0, 1]
        for size in (1, 2, 5):
            with self.subTest(size=size):
                strategy = self.qs_class(cognition_window_size=size, **kwargs)
                for i, candidate in enumerate(candidates):
                    strategy.update([candidate], [])
                    for name in (
                        "cognition_window_",
                        "min_dist_",
                        "theta_",
                        "s_",
                        "t_x_",
                        "f_",
                    ):
                        self.assertEqual(
                            len(getattr(strategy, name)), min(i + 1, size)
                        )
                batch_strategy = self.qs_class(
                    cognition_window_size=size, **kwargs
                )
                batch_strategy.update(candidates, [])
                _assert_acquisition_state_equal(self, batch_strategy, strategy)

    def test_init_param_density_threshold(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, TypeError),
            (-1, ValueError),
            (1, None),
        ]
        self._test_param("init", "density_threshold", test_cases)

    def test_init_param_cognition_window_size(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (10, None),
        ]
        self._test_param("init", "cognition_window_size", test_cases)

    def test_init_param_dist_func(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (pairwise_distances, None),
            (None, None),
        ]
        self._test_param("init", "dist_func", test_cases)
        replace_init_params = {"dist_func": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )
        replace_init_params = {"dist_func": pairwise_distances}
        test_cases = [(np.array([[1, 2]]), None)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_init_param_dist_func_dict(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (["func"], TypeError),
            ({"metric": "manhattan"}, None),
        ]
        self._test_param("init", "dist_func_dict", test_cases)
        replace_init_params = {"dist_func_dict": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_init_param_force_full_budget(self):
        test_cases = []
        test_cases += [("string", TypeError), (True, None), (False, None)]
        self._test_param("init", "force_full_budget", test_cases)

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB()), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)


class TestCognitiveDualQueryStrategy(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
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
            qs_class=CognitiveDualQueryStrategy,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_batch_accounts_for_earlier_acquisitions(self):
        _assert_batch_accounts_for_earlier_acquisitions(self, self.qs_class)

    def test_evicted_samples_do_not_count_towards_density(self):
        strategy = CognitiveDualQueryStrategy(
            cognition_window_size=1,
            budget_manager=RecordingBudgetManager(),
            random_state=0,
        )
        candidates = np.array([[0.0], [10.0], [0.0]])
        strategy.update(candidates, [], {"utilities": np.ones(3)})
        history = strategy.budget_manager_.history_
        self.assertEqual([entry[0].tolist() for entry in history], [[10.0]])

    def test_update_accepts_its_own_filtered_query_indices(self):
        classifier = _empty_classifier()
        strategy = CognitiveDualQueryStrategy(
            budget_manager=FixedUncertaintyBudgetManager([0, 1], budget=0.1),
            random_state=0,
        )
        candidates = np.array([[0.0], [0.5]])
        queried = strategy.query(candidates, classifier)
        self.assertEqual(queried, [1])
        self.assertIs(strategy.update(candidates, queried), strategy)
        self.assertEqual(strategy.budget_manager_.u_t_, 1)
        self.assertEqual(strategy.t_, 2)

    def test_filtered_updates_align_utilities_and_actual_acquisitions(self):
        candidates = np.array([0.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.875])[
            :, None
        ]
        classifier = _empty_classifier()
        for full in (False, True):
            with self.subTest(full=full):
                strategy = CognitiveDualQueryStrategy(
                    force_full_budget=full,
                    budget_manager=RecordingBudgetManager(),
                    random_state=0,
                )
                queried, utilities = strategy.query(
                    candidates, classifier, return_utilities=True
                )
                self.assertEqual(queried, [2] if full else [1, 5])
                self.assertEqual(strategy.budget_manager_.history_, [])
                supplied_utilities = np.arange(8, dtype=float)
                params = {
                    "utilities": supplied_utilities,
                    "marker": "kept",
                }
                acquired = 2 if full else 5
                strategy.update(candidates, [acquired], params)
                history = strategy.budget_manager_.history_
                expected_indices = list(range(8)) if full else [1, 2, 5]
                self.assertEqual(
                    [entry[2] for entry in history], expected_indices
                )
                self.assertEqual(
                    [entry[1] for entry in history],
                    [i == acquired for i in expected_indices],
                )
                self.assertTrue(all(entry[3] == "kept" for entry in history))
                self.assertEqual(strategy.t_, len(candidates))
                self.assertIs(params["utilities"], supplied_utilities)
                np.testing.assert_array_equal(supplied_utilities, np.arange(8))

    def test_invalid_indices_do_not_advance_state(self):
        _assert_invalid_indices_do_not_advance_state(self, self.qs_class)

    def test_query_isolates_global_manager_rng_and_preserves_state(self):
        _assert_query_isolates_rng_and_preserves_state(self, self.qs_class)

    def test_query_failures_restore_density_and_budget_state(self):
        _assert_query_failures_restore_state(self, self.qs_class)

    def test_no_eligible_candidates_and_zero_density_threshold(self):
        candidates = np.zeros((6, 1))
        classifier = _empty_classifier()
        for full in (False, True):
            for threshold in (0, 100):
                with self.subTest(full=full, threshold=threshold):
                    strategy = CognitiveDualQueryStrategy(
                        force_full_budget=full,
                        density_threshold=threshold,
                        budget_manager=RecordingBudgetManager(),
                        random_state=0,
                    )
                    queried, utilities = strategy.query(
                        candidates, classifier, return_utilities=True
                    )
                    self.assertEqual(
                        queried, [0, 2, 4] if threshold == 0 else []
                    )
                    np.testing.assert_array_equal(utilities, np.full(6, 0.5))
                    strategy.update(
                        candidates, queried, {"utilities": utilities}
                    )
                    self.assertEqual(strategy.t_, 6)
                    self.assertEqual(
                        len(strategy.budget_manager_.history_),
                        6 if full or threshold == 0 else 0,
                    )

    def test_query(self):
        expected_output = [4, 5, 6, 7, 11]
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


class TestCognitiveDualQueryStrategyVarUn(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
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
            qs_class=CognitiveDualQueryStrategyVarUn,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = [4, 5, 6, 7]
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


class TestCognitiveDualQueryStrategyRanVarUn(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
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
            qs_class=CognitiveDualQueryStrategyRanVarUn,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = [4, 5, 6, 7, 11]
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


class TestCognitiveDualQueryStrategyRan(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
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
            qs_class=CognitiveDualQueryStrategyRan,
            init_default_params={"force_full_budget": True},
            query_default_params_clf=query_default_params_clf,
        )

    def test_query(self):
        expected_output = [11]
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


class TestCognitiveDualQueryStrategyFixUn(
    TemplateCognitiveDualQueryStrategy, unittest.TestCase
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
            qs_class=CognitiveDualQueryStrategyFixUn,
            init_default_params={
                "force_full_budget": True,
                "classes": self.classes,
            },
            query_default_params_clf=query_default_params_clf,
        )

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

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, TypeError),
            (CognitiveDualQueryStrategyFixUn, TypeError),
            ([[0, 1], [0, 1]], ValueError),
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


class TestStreamDensityBasedAL(
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
            qs_class=StreamDensityBasedAL,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_batch_accounts_for_earlier_acquisitions(self):
        _assert_batch_accounts_for_earlier_acquisitions(self, self.qs_class)

    def test_partition_invariance(self):
        _assert_partition_invariance(self, self.qs_class)

    def test_dist_func_dict_is_forwarded(self):
        _assert_dist_func_dict_is_forwarded(self, self.qs_class)

    def test_invalid_indices_do_not_advance_state(self):
        _assert_invalid_indices_do_not_advance_state(self, self.qs_class)

    def test_query_isolates_global_manager_rng_and_preserves_state(self):
        _assert_query_isolates_rng_and_preserves_state(self, self.qs_class)

    def test_query_failures_restore_density_and_budget_state(self):
        _assert_query_failures_restore_state(self, self.qs_class)

    def test_no_eligible_candidates(self):
        candidates = np.zeros((6, 1))
        classifier = _empty_classifier()
        strategy = StreamDensityBasedAL(
            budget_manager=RecordingBudgetManager(), random_state=0
        )
        strategy.update(candidates[:2], [], {"utilities": np.ones(2)})
        queried, utilities = strategy.query(
            candidates, classifier, return_utilities=True
        )
        self.assertEqual(queried, [])
        strategy.update(candidates, queried, {"utilities": utilities})
        self.assertEqual(len(strategy.budget_manager_.history_), 8)

    def test_init_param_window_size(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (0.0, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (100, None),
        ]
        self._test_param("init", "window_size", test_cases)

    def test_init_param_dist_func(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (pairwise_distances, None),
            (None, None),
        ]
        self._test_param("init", "dist_func", test_cases)
        replace_init_params = {"dist_func": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )
        replace_init_params = {"dist_func": pairwise_distances}
        test_cases = [(np.array([[1, 2]]), None)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_init_param_dist_func_dict(self):
        test_cases = []
        test_cases += [
            ("string", TypeError),
            (["func"], TypeError),
            ({"metric": "manhattan"}, None),
        ]
        self._test_param("init", "dist_func_dict", test_cases)
        replace_init_params = {"dist_func_dict": []}
        test_cases = [(np.array([[1, 2]]), TypeError)]
        self._test_param(
            "update",
            "candidates",
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianNB(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(GaussianNB(), classes=[0, 1]), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query(self):
        expected_output = []
        expected_utilities = [
            3.2717822e-04,
            4.8216976e-05,
            3.4917266e-08,
            2.4212931e-03,
            2.2640944e-03,
            3.5978039e-02,
            3.1426270e-01,
            6.8819578e-02,
            1.8494024e-01,
            3.9211023e-02,
            1.3801839e-02,
            1.1915575e-04,
            3.6804321e-03,
            1.0421308e-04,
            5.6003508e-03,
            5.1317648e-03,
        ]
        return super().test_query(expected_output, expected_utilities)

    def test_query_single_class(self):
        X = np.arange(8, dtype=float).reshape(-1, 2)
        y = np.array([0.0, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL])
        clf = ParzenWindowClassifier(random_state=0).fit(X, y)

        _, utilities = StreamDensityBasedAL(random_state=0).query(
            X[1:],
            clf,
            X=X,
            y=y,
            return_utilities=True,
        )

        np.testing.assert_array_equal(utilities, np.zeros(3))
