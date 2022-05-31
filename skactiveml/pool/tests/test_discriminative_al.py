import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import DiscriminativeAL


class TestDiscriminativeAL(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.y_unlblb = np.full_like(self.y, -1)
        self.X = StandardScaler().fit_transform(self.X)
        self.discriminator = ParzenWindowClassifier(
            random_state=self.random_state
        )

    def test_init_param_greedy_selection(self):
        for greedy_selection in [0, "test", None]:
            dal = DiscriminativeAL(
                greedy_selection=greedy_selection, missing_label=-1
            )
            self.assertRaises(
                TypeError,
                dal.query,
                X=self.X,
                y=self.y_unlblb,
                discriminator=self.discriminator,
            )

    def test_query_param_discriminator(self):
        dal = DiscriminativeAL(missing_label=-1)
        for discriminator in [None, GaussianProcessClassifier(), "test"]:
            self.assertRaises(
                TypeError,
                dal.query,
                X=self.X,
                y=self.y_unlblb,
                discriminator=discriminator,
            )

    def test_query(self):
        for greedy_selection in [False, True]:
            dal = DiscriminativeAL(
                missing_label=-1,
                random_state=self.random_state,
                greedy_selection=greedy_selection,
            )
            for candidates in [None, np.arange(len(self.X))]:
                if candidates is None:
                    n_candidates = len(self.y_unlblb)
                else:
                    n_candidates = len(candidates)
                query_indices = dal.query(
                    X=self.X,
                    y=self.y_unlblb,
                    discriminator=self.discriminator,
                    candidates=candidates,
                )
                self.assertEqual(1, len(query_indices))
                query_indices, utilities = dal.query(
                    X=self.X,
                    y=self.y_unlblb,
                    discriminator=self.discriminator,
                    return_utilities=True,
                    candidates=candidates,
                )
                self.assertEqual(self.discriminator.classes, None)
                self.assertFalse(hasattr(self.discriminator, "classes_"))
                self.assertEqual(1, len(query_indices))
                np.testing.assert_array_equal(
                    np.ones((1, n_candidates)), utilities
                )
                query_indices, utilities = dal.query(
                    X=self.X,
                    y=self.y_unlblb,
                    discriminator=self.discriminator,
                    candidates=candidates,
                    return_utilities=True,
                    batch_size=10,
                )
                self.assertEqual(10, len(query_indices))
                for i in range(10):
                    self.assertEqual(i, np.sum(np.isnan(utilities[i])))
                    if greedy_selection:
                        default_utilities = np.ones(n_candidates)
                        is_nan = np.isnan(utilities[i])
                        np.testing.assert_array_equal(
                            default_utilities[~is_nan], utilities[i, ~is_nan]
                        )
                    if not greedy_selection and i < 9:
                        is_nan = np.isnan(utilities[i + 1])
                        self.assertRaises(
                            AssertionError,
                            np.testing.assert_array_equal,
                            utilities[i, ~is_nan],
                            utilities[i + 1, ~is_nan],
                        )
