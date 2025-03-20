import unittest
import warnings

import numpy as np

from skactiveml.stream.budgetmanager import SplitBudgetManager
from skactiveml.utils import (
    check_cost_matrix,
    check_classes,
    check_scalar,
    check_X_y,
    check_type,
    check_bound,
    check_budget_manager,
    check_classifier_params,
    check_indices,
    check_n_features,
    check_random_state,
    check_class_prior,
)
from skactiveml.utils._validation import _check_callable


class TestValidation(unittest.TestCase):
    def test_check_scalar(self):
        x = 5
        self.assertRaises(
            TypeError, check_scalar, x=x, target_type=float, name="x"
        )
        self.assertRaises(
            ValueError, check_scalar, x=x, target_type=int, max_val=4, name="x"
        )
        self.assertRaises(
            ValueError,
            check_scalar,
            x=x,
            target_type=int,
            max_inclusive=False,
            max_val=5,
            name="x",
        )
        self.assertRaises(
            ValueError, check_scalar, x=x, target_type=int, min_val=6, name="x"
        )
        self.assertRaises(
            ValueError,
            check_scalar,
            x=x,
            target_type=int,
            min_inclusive=False,
            min_val=5,
            name="x",
        )
        self.assertRaises(
            ValueError,
            check_scalar,
            x=np.nan,
            target_type=float,
            min_inclusive=False,
            min_val=5,
            name="x",
        )
        self.assertRaises(
            ValueError,
            check_scalar,
            x=np.nan,
            target_type=float,
            min_inclusive=False,
            max_val=5,
            name="x",
        )
        check_scalar(x=np.nan, name="x", target_type=float)

    def test_check_classifier_params(self):
        self.assertRaises(
            ValueError,
            check_classifier_params,
            classes=[0, 1, 2],
            missing_label=np.nan,
            cost_matrix=[[1, 1], [2, 0]],
        )
        self.assertRaises(
            TypeError,
            check_classifier_params,
            classes=["a", "b"],
            missing_label=2,
            cost_matrix=[[1, 1], [2, 0]],
        )
        self.assertRaises(
            TypeError,
            check_classifier_params,
            classes=[0, 1],
            missing_label="nan",
            cost_matrix=[[1, 1], [2, 0]],
        )
        self.assertRaises(
            ValueError,
            check_classifier_params,
            classes=None,
            missing_label=np.nan,
            cost_matrix=[[1, 1], [2, 0]],
        )
        self.assertRaises(
            ValueError,
            check_classifier_params,
            classes=[np.nan, 1],
            missing_label=np.nan,
        )
        self.assertRaises(
            ValueError,
            check_classifier_params,
            classes=[-1, 1],
            missing_label=-1,
        )
        self.assertRaises(
            ValueError,
            check_classifier_params,
            classes=["a", "b"],
            missing_label="a",
        )

    def test_check_classes(self):
        self.assertRaises(TypeError, check_classes, classes=[None, 1, 2])
        self.assertRaises(TypeError, check_classes, classes=["2", 1, 2])
        self.assertRaises(TypeError, check_classes, classes=2)
        self.assertRaises(ValueError, check_classes, classes=[1, 2, 2])

    def test_check_class_prior(self):
        self.assertRaises(TypeError, check_class_prior, None, 1)
        self.assertRaises(TypeError, check_class_prior, 1, None)
        self.assertRaises(ValueError, check_class_prior, 1, 0)
        self.assertRaises(ValueError, check_class_prior, -2, 2)
        self.assertRaises(ValueError, check_class_prior, [0, 1, -1], 3)
        self.assertRaises(ValueError, check_class_prior, [1, 2, 3], 2)
        np.testing.assert_array_equal(check_class_prior(1, 3), [1, 1, 1])

    def test_check_cost_matrix(self):
        self.assertRaises(
            ValueError,
            check_cost_matrix,
            cost_matrix=[["2", "5"], ["a", "5"]],
            n_classes=2,
        )
        self.assertRaises(
            ValueError,
            check_cost_matrix,
            cost_matrix=[[0, 1], [2, 0]],
            n_classes=3,
        )
        self.assertRaises(
            ValueError,
            check_cost_matrix,
            cost_matrix=[[0, 1], [2, 0]],
            n_classes=-1,
        )
        self.assertRaises(
            TypeError,
            check_cost_matrix,
            cost_matrix=[[0, 1], [2, 0]],
            n_classes=2.5,
        )
        self.assertRaises(
            ValueError,
            check_cost_matrix,
            cost_matrix=[[2, 1], [2, 2]],
            n_classes=2,
            diagonal_is_zero=True,
        )
        self.assertRaises(
            ValueError,
            check_cost_matrix,
            cost_matrix=[[0, 1], [-1, 0]],
            n_classes=2,
            only_non_negative=True,
        )
        self.assertRaises(
            ValueError,
            check_cost_matrix,
            cost_matrix=[[0, 0], [0, 0]],
            n_classes=2,
            contains_non_zero=True,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_cost_matrix(cost_matrix=[[1, 1], [2, 0]], n_classes=2)
            check_cost_matrix(cost_matrix=[[0, 1], [-1, 0]], n_classes=2)
            check_cost_matrix(cost_matrix=[[0, 0], [0, 0]], n_classes=2)
            assert len(w) == 3

    def test_check_X_y(self):
        X = [[1, 2], [3, 4]]
        y = [1, 0]
        X_cand = [[5, 6]]
        X, y, sample_weight = check_X_y(X, y)
        np.testing.assert_array_equal(sample_weight, np.array([1.0, 1.0]))
        X, y, X_cand, sample_weight, sample_weight_cand = check_X_y(
            X, y, X_cand
        )
        np.testing.assert_array_equal(sample_weight_cand, np.array([1.0]))
        sample_weight = [0.4, 0.6]
        X, y, X_cand, sample_weight, _ = check_X_y(X, y, X_cand, sample_weight)
        self.assertTrue(isinstance(X, np.ndarray))
        y = [[1], [0]]
        X, y, X_cand, sample_weight, _ = check_X_y(
            X, y, X_cand, sample_weight, multi_output=True
        )
        self.assertTrue(isinstance(y, np.ndarray))
        y = np.array([1, 0], dtype=object)
        X, y, X_cand, sample_weight, _ = check_X_y(
            X, y, X_cand, sample_weight, y_numeric=True
        )
        X_cand_false = [[5]]
        self.assertRaises(
            ValueError,
            check_X_y,
            X,
            y,
            X_cand_false,
            sample_weight,
            multi_output=True,
        )
        y = np.array([[1, 0, 1], [2, 0, 1]])
        self.assertRaises(
            ValueError,
            check_X_y,
            X,
            y,
            X_cand,
            sample_weight,
            multi_output=True,
        )

    def test_check_random_state(self):
        seed = 12
        self.assertRaises(ValueError, check_random_state, "string")
        self.assertRaises(TypeError, check_random_state, seed, "string")

        random_state = np.random.RandomState(seed)
        ra = check_random_state(random_state, 3)
        rb = check_random_state(random_state, 3)
        self.assertTrue(ra.rand() == rb.rand())

        ra = check_random_state(42, 3)
        rb = check_random_state(42, 3)
        self.assertTrue(ra.rand() == rb.rand())

        ra = check_random_state(None)
        rb = check_random_state(None)
        self.assertTrue(ra.rand() != rb.rand())
        ra = check_random_state(np.random.RandomState(None))
        rb = check_random_state(np.random.RandomState(None))
        self.assertTrue(ra.rand() != rb.rand())

    def test_check_type(self):
        self.assertRaises(TypeError, check_type, 10, "a", str)
        self.assertRaises(TypeError, check_type, 10, "a", str, bool)
        self.assertRaises(TypeError, check_type, 10, "a", str, bool, map, list)
        check_type(10, "a", int)
        check_type("number", "a", target_vals=["number"])
        self.assertRaises(
            TypeError, check_type, "number", "a", target_vals=["nonumber"]
        )
        self.assertRaises(
            TypeError, check_type, 10, "a", str, target_vals=[12]
        )

        def is_prime(x):
            return (
                len(list(filter(lambda n: x % n == 0, range(1, x + 1)))) == 2
            )

        self.assertRaises(
            TypeError, check_type, 10, "a", indicator_funcs=[is_prime]
        )
        self.assertRaises(
            TypeError, check_type, 10, "a", str, indicator_funcs=[is_prime]
        )
        check_type(7, "a", str, indicator_funcs=[is_prime])
        self.assertRaises(
            TypeError,
            check_type,
            10,
            "a",
            dict,
            target_vals=["undefined"],
            indicator_funcs=[is_prime],
        )

    def test_check_callable(self):
        self.assertRaises(
            ValueError,
            _check_callable,
            lambda x: x,
            "name",
            n_positional_parameters=2,
        )

        self.assertRaises(
            ValueError,
            _check_callable,
            lambda x, y: x,
            "name",
            n_positional_parameters=1,
        )

        self.assertRaises(TypeError, _check_callable, "illegal", "name")

    def test_check_indices_single_dimension(self):
        A = np.array([[4, 5], [6, 1], [3, 4]])
        ind = np.array([0, 2])
        self.assertRaises(ValueError, check_indices, "a", ind)
        self.assertRaises(TypeError, check_indices, 42, ind)
        self.assertRaises(TypeError, check_indices, A, "b")
        self.assertRaises(TypeError, check_indices, A, 7)
        self.assertRaises(ValueError, check_indices, A, ind, dim=2)
        ind_out_of_range = np.array([0, 3])
        self.assertRaises(ValueError, check_indices, ind_out_of_range, A)
        ind_not_unique = np.array([0, 0])
        self.assertRaises(
            ValueError, check_indices, ind_not_unique, A, unique="check_unique"
        )
        indices_now_unique = check_indices(ind_not_unique, A, unique=True)
        self.assertEqual(len(indices_now_unique), 1)

    def test_check_indices_n_dimensions(self):
        A = np.array([[4, 5], [6, 1], [3, 4]])
        ind = np.array([[0, 1], [2, 0]])
        self.assertRaises(ValueError, check_indices, A, ind, dim=(0, 1, 2))
        self.assertRaises(ValueError, check_indices, A, ind, dim=(0,))
        self.assertRaises(TypeError, check_indices, A, ind, dim=(0, (0, 1)))
        ind_out_of_range = np.array([[0, 3]])
        self.assertRaises(ValueError, check_indices, ind_out_of_range, A)
        ind_not_unique = np.array([[0, 0], [0, 0]])
        self.assertRaises(
            ValueError, check_indices, ind_not_unique, A, unique="check_unique"
        )
        indices_now_unique = check_indices(ind_not_unique, A, unique=True)
        self.assertEqual(2, len(indices_now_unique))
        for i in [0, 1]:
            self.assertEqual(1, len(indices_now_unique[i]))

    def test_check_bound(self):
        self.assertRaises(ValueError, check_bound, X=7)
        self.assertRaises(ValueError, check_bound, bound=7)

        X = np.array([[3, 4], [2, 7], [-1, 5]])
        wrong_X = np.array([[3, 4, 2]])
        correct_bound = np.array([[-1, 4], [3, 7]])
        small_bound = np.array([[1, 4], [3, 7]])
        wrong_bound = np.array([[1, 4]])

        re_correct_bound = check_bound(bound=correct_bound, X=X)
        re_no_bound = check_bound(X=X)
        re_no_X = check_bound(bound=correct_bound)
        np.testing.assert_array_equal(correct_bound, re_correct_bound)
        np.testing.assert_array_equal(correct_bound, re_no_bound)
        np.testing.assert_array_equal(correct_bound, re_no_X)
        with self.assertWarns(Warning):
            check_bound(small_bound, re_correct_bound)
        self.assertRaises(ValueError, check_bound, X=wrong_X)
        self.assertRaises(ValueError, check_bound, bound=wrong_bound)
        self.assertRaises(ValueError, check_bound)
        self.assertRaises(
            ValueError, check_bound, X=X, bound_must_be_given=True
        )

    def test_check_budget_manager(self):
        self.assertIsNotNone(
            check_budget_manager(0.1, None, SplitBudgetManager)
        )
        with self.assertWarns(Warning):
            check_budget_manager(
                0.1, SplitBudgetManager(budget=0.2), SplitBudgetManager
            )

    def test_check_n_features(self):

        # Define a simple DummyEstimator
        class DummyEstimator:
            def __init__(self, n_features_in_=None):
                self.n_features_in_ = n_features_in_

        # Test 1: reset=True with non-empty X
        with self.subTest("reset True with non-empty X"):
            est = DummyEstimator()
            X = np.array([[1, 2, 3], [4, 5, 6]])
            check_n_features(est, X, reset=True)
            self.assertEqual(est.n_features_in_, 3)

        # Test 2: reset=True with empty X
        with self.subTest("reset True with empty X"):
            est = DummyEstimator()
            X = np.empty((0, 3))
            check_n_features(est, X, reset=True)
            self.assertIsNone(est.n_features_in_)

        # Test 3: reset=False when n_features_in_ is None (no check)
        with self.subTest("reset False without n_features_in_ set"):
            est = DummyEstimator()
            X = np.array([[1, 2, 3], [4, 5, 6]])
            check_n_features(est, X, reset=False)
            self.assertIsNone(est.n_features_in_)

        # Test 4: reset=False with correct feature count
        with self.subTest("reset False with matching features"):
            est = DummyEstimator(n_features_in_=3)
            X = np.array([[1, 2, 3], [7, 8, 9]])
            try:
                check_n_features(est, X, reset=False)
            except ValueError:
                self.fail("check_n_features raised ValueError unexpectedly!")

        # Test 5: reset=False with incorrect feature count
        with self.subTest("reset False with mismatched features"):
            est = DummyEstimator(n_features_in_=3)
            X = np.array([[1, 2], [3, 4]])
            with self.assertRaises(ValueError):
                check_n_features(est, X, reset=False)
