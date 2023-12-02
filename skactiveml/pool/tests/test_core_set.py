import unittest

import numpy as np
from skactiveml.pool._core_set import CoreSet
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

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, ValueError),
            ("string", ValueError),
            (None, ValueError),
            ("greedy", None),
        ]
        self._test_param("init", "method", test_cases)

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
                    self.assertGreaterEqual(i, 0)
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

        # test case 5: for candidates with new samples
        X_cand = random_state.choice(5, size=(5, 2))
        _, utilities_5 = core_set_1.query(
            X, y_1, batch_size=2, candidates=X_cand, return_utilities=True
        )
        self.assertEqual(5, utilities_5.shape[1])




