import unittest

import numpy as np
from sklearn.svm import SVC

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import DiscriminativeAL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestDiscriminativeAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.random_state = 1
        self.X = np.linspace(0, 1, 20).reshape(10, 2)
        self.y = np.hstack([[0, 1], np.full(8, MISSING_LABEL)])
        self.y_reg = np.hstack([[0.5, 1.6], np.full(8, MISSING_LABEL)])
        self.discriminator = ParzenWindowClassifier(
            random_state=self.random_state
        )
        query_default_params_clf = {
            "X": self.X,
            "y": self.y,
            "discriminator": self.discriminator,
        }
        query_default_params_reg = {
            "X": self.X,
            "y": self.y_reg,
            "discriminator": self.discriminator,
        }
        super().setUp(
            qs_class=DiscriminativeAL,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_greedy_selection(self):
        test_cases = [
            (0, TypeError),
            ("test", TypeError),
            (None, TypeError),
            (SVC(), TypeError),
        ]
        self._test_param("init", "greedy_selection", test_cases)

    def test_query_param_discriminator(self):
        test_cases = [
            (0, TypeError),
            ("test", TypeError),
            (None, TypeError),
            (SVC(), TypeError),
        ]
        self._test_param("query", "discriminator", test_cases)

    def test_query(self):
        for greedy_selection in [False, True]:
            dal = DiscriminativeAL(
                random_state=self.random_state,
                greedy_selection=greedy_selection,
            )
            for candidates in [None, np.arange(len(self.X))]:
                if candidates is None:
                    n_candidates = len(self.y)
                else:
                    n_candidates = len(candidates)
                query_indices = dal.query(
                    X=self.X,
                    y=self.y,
                    discriminator=self.discriminator,
                    candidates=candidates,
                )
                self.assertEqual(1, len(query_indices))
                query_indices, utilities = dal.query(
                    X=self.X,
                    y=np.full_like(self.y, MISSING_LABEL),
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
                    y=np.full_like(self.y, MISSING_LABEL),
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
