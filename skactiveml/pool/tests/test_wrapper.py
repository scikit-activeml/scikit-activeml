import unittest

import numpy as np

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.pool._wrapper import UtilityWrapper


class TestUtilityWrapper(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = ParzenWindowClassifier()
        self.kwargs = dict(
            X=self.X, y=self.y, candidates=self.candidates, clf=self.clf
        )

    def test_init_param_query_strategy(self):
        self.assertRaises(TypeError, UtilityWrapper, 'String')

    def test_query_param_utility_weights(self):
        qs = UtilityWrapper(UncertaintySampling())
        self.assertRaises(ValueError,
                          qs.query, **self.kwargs, utility_weights='String')

    def test_query(self):
        qs = UtilityWrapper(UncertaintySampling())
        qs.query(**self.kwargs, utility_weights=[1, 0, 0])
        for i in range(len(self.candidates)):
            uw = np.zeros(shape=len(self.candidates))
            uw[i] = 1
            idx = qs.query(**self.kwargs, utility_weights=uw)
            self.assertEqual(idx[0], i)

        # test utilities
        utility_weights = np.random.rand(len(self.candidates))
        qs = UtilityWrapper(UncertaintySampling())
        _, utils_w = qs.query(**self.kwargs, utility_weights=utility_weights,
                           return_utilities=True)
        qs = UncertaintySampling()
        _, utils = qs.query(**self.kwargs, return_utilities=True)
        np.testing.assert_array_equal(utils*utility_weights, utils_w)


        # random_state
        cand = np.ones(shape=(100, 2))
        kwargs = self.kwargs
        kwargs['candidates'] = cand
        random_state = 42
        qs = UncertaintySampling(random_state=random_state)
        qs_w = UtilityWrapper(UncertaintySampling(random_state=random_state))
        idx_w = qs_w.query(**self.kwargs, batch_size=len(cand), utility_weights=np.ones(shape=len(cand)))
        idx = qs.query(**self.kwargs, batch_size=len(cand))

        np.testing.assert_array_equal(idx_w, idx)

    def test_attr(self):
        qs = UncertaintySampling()
        attributes = qs.__dict__
        qs = UtilityWrapper(qs)
        for attr in attributes:
            getattr(qs, attr)
