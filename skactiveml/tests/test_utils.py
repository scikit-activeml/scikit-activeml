import unittest
from copy import deepcopy

import numpy as np
from sklearn.naive_bayes import GaussianNB

from skactiveml.tests.utils import state_difference


class TestStateDifference(unittest.TestCase):
    def setUp(self):
        X = np.arange(20.0).reshape(10, 2)
        y = np.array([0] * 5 + [1] * 5)
        self.clf = GaussianNB().fit(X, y)

    def test_equal_states(self):
        for state in [
            {"coef": 1},
            {"coef": np.array([[1.0, np.nan]])},
            {"label": "before"},
            {"nested": {"coef": [1, (2, 3)]}},
            {"classes": np.array(["a", "b"])},
            {"random_state": np.random.RandomState(0)},
            {"clf": self.clf},
        ]:
            with self.subTest(state=list(state)[0]):
                self.assertIsNone(state_difference(state, deepcopy(state)))

    def test_changed_scalar(self):
        self.assertIsNotNone(state_difference({"coef": 1}, {"coef": 999}))

    def test_changed_array_element(self):
        self.assertIsNotNone(
            state_difference(
                {"coef": np.array([1.0, 2.0])},
                {"coef": np.array([1.0, 999.0])},
            )
        )

    def test_changed_nan_placement(self):
        self.assertIsNotNone(
            state_difference(
                {"coef": np.array([np.nan, 2.0])},
                {"coef": np.array([2.0, np.nan])},
            )
        )

    def test_changed_string(self):
        self.assertIsNotNone(
            state_difference({"label": "before"}, {"label": "after"})
        )

    def test_changed_class_labels(self):
        self.assertIsNotNone(
            state_difference(
                {"classes": np.array(["a", "b"])},
                {"classes": np.array(["a", "c"])},
            )
        )

    def test_changed_nested_estimator(self):
        changed = deepcopy(self.clf)
        changed.theta_[0, 0] += 1
        self.assertIsNotNone(
            state_difference({"clf": self.clf}, {"clf": changed})
        )

    def test_changed_sequence_element(self):
        self.assertIsNotNone(
            state_difference({"coef": [1, 2]}, {"coef": [1, 999]})
        )

    def test_changed_type(self):
        self.assertIsNotNone(state_difference({"coef": 1}, {"coef": 1.0}))

    def test_missing_key(self):
        self.assertIsNotNone(
            state_difference({"coef": 1}, {"coef": 1, "intercept": 0})
        )

    def test_added_key(self):
        self.assertIsNotNone(
            state_difference({"coef": 1, "intercept": 0}, {"coef": 1})
        )

    def test_advanced_random_state(self):
        expected = np.random.RandomState(0)
        actual = deepcopy(expected)
        actual.random_sample(1)
        self.assertIsNotNone(state_difference(actual, expected))

    def test_difference_names_the_path(self):
        difference = state_difference(
            {"nested": {"coef": np.array([1.0, 2.0])}},
            {"nested": {"coef": np.array([1.0, 999.0])}},
            name="clf",
        )
        self.assertIn("clf.nested.coef[1]", difference)


if __name__ == "__main__":
    unittest.main()
