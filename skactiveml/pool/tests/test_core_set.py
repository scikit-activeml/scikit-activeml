import unittest

import numpy as np
from skactiveml.pool._core_set import CoreSet, k_greedy_center
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestCoreSet(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        query_default_params = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.hstack([[0, 1], np.full(8, MISSING_LABEL)]),
        }
        super().setUp(
            qs_class=CoreSet,
            init_default_params={},
            query_default_params_clf=query_default_params,
        )

    def test_query(self):
        # test case 1: with the same random state the init pick up
        # is the same
        core_set_1 = CoreSet(random_state=42)
        random_state = np.random.RandomState(42)

        X = random_state.choice(5, size=(10, 2))
        y = np.full(10, MISSING_LABEL)

        self.assertEqual(core_set_1.query(X, y), core_set_1.query(X, y))

        # test case 2: all utilities are not negative or np.nan
        y_1 = np.hstack([[0], np.full(9, MISSING_LABEL)])
        _, utilities = core_set_1.query(
            X, y_1, batch_size=2, return_utilities=True
        )
        for u in utilities:
            for i in u:
                if not np.isnan(i):
                    self.assertGreaterEqual(i, 0)
                else:
                    self.assertTrue(np.isnan(i))

        # test case 3: all samples have the same features, the utilities
        # are also the same.
        X_3 = np.ones((10, 2))
        y_3 = np.hstack([[0, 1], np.full(8, MISSING_LABEL)])

        _, utilities_3 = core_set_1.query(
            X_3, y_3, batch_size=1, return_utilities=True
        )
        for u in utilities_3:
            for i in u:
                if not np.isnan(i):
                    self.assertEqual(i, 0)
                else:
                    self.assertTrue(np.isnan(i))

        # test case 4: for candidates.ndim = 1
        candidates = np.arange(1, 5)
        _, utilities_4 = core_set_1.query(
            X, y_1, batch_size=1, candidates=candidates, return_utilities=True
        )
        for u in utilities_4:
            for idx, value in enumerate(u):
                if idx in candidates:
                    self.assertGreaterEqual(value, 0)
                else:
                    self.assertTrue(np.isnan(value))
        self.assertEqual(len(X), utilities_4.shape[1])

        # test case 5: for candidates with new samples
        X_cand = random_state.choice(5, size=(5, 2))
        _, utilities_5 = core_set_1.query(
            X, y_1, batch_size=2, candidates=X_cand, return_utilities=True
        )
        self.assertEqual(5, utilities_5.shape[1])

        # test case 6: remove already unlabeled samples `X[unlbld_indices]`
        # from `X` and add them as `candidates=X[unlbld_indices]` and check
        # whether the utilities are the same.
        core_set_6 = CoreSet(random_state=42)

        X = random_state.choice(5, size=(10, 2))
        y = np.full(10, MISSING_LABEL)
        y[:5] = 0

        query_indicies_6_1, utilities_6_1 = core_set_6.query(
            X, y, candidates=None, return_utilities=True
        )
        query_indicies_6_2, utilities_6_2 = core_set_6.query(
            X, y, candidates=[5, 6, 7, 8, 9], return_utilities=True
        )
        _, utilities_6_3 = core_set_6.query(
            X, y, candidates=X[[5, 6, 7, 8, 9]], return_utilities=True
        )

        np.testing.assert_array_equal(query_indicies_6_1, query_indicies_6_2)
        np.testing.assert_array_equal(utilities_6_1, utilities_6_2)
        np.testing.assert_array_equal(utilities_6_1[:, 5:], utilities_6_3)

        # test case 7: initial selection
        y_7 = np.full(10, MISSING_LABEL)
        _, utilities = core_set_1.query(
            X, y_7, batch_size=2, return_utilities=True
        )


class TestKGreedyCenter(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(42).choice(5, size=(10, 2))
        self.y = np.hstack([[0], np.full(9, MISSING_LABEL)])
        self.batch_size = (1,)
        self.random_state = None
        self.missing_label = (np.nan,)
        self.mapping = None
        self.n_new_cand = None

    def test_param_X(self):
        self.assertRaises(ValueError, k_greedy_center, X=[1], y=[np.nan])
        self.assertRaises(ValueError, k_greedy_center, X="string", y=[np.nan])

    def test_param_y(self):
        self.assertRaises(TypeError, k_greedy_center, X=[[1, 1]], y=1)

    def test_param_batch_size(self):
        self.assertRaises(
            TypeError, k_greedy_center, X=self.X, y=self.y, batch_size="string"
        )

    def test_param_random_state(self):
        self.assertRaises(
            ValueError,
            k_greedy_center,
            X=self.X,
            y=self.y,
            random_state="string",
        )

    def test_param_mapping(self):
        self.assertRaises(
            ValueError, k_greedy_center, X=self.X, y=self.y, mapping="string"
        )

    def test_param_n_new_cand(self):
        self.assertRaises(
            TypeError, k_greedy_center, X=self.X, y=self.y, n_new_cand="string"
        )
        self.assertRaises(
            ValueError,
            k_greedy_center,
            X=self.X,
            y=self.y,
            mapping=np.arange(4),
            n_new_cand=5,
        )

    def test_k_greedy_center(self):
        random_state = np.random.RandomState(42)
        query_idx_1, _ = k_greedy_center(self.X, self.y, random_state=42)
        query_idx_2, _ = k_greedy_center(
            self.X, self.y, random_state=random_state
        )
        self.assertEqual(query_idx_1, query_idx_2)
