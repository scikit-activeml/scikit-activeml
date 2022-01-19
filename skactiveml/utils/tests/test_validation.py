import unittest
import warnings

import numpy as np

from skactiveml.utils import check_cost_matrix, check_classes, \
    check_missing_label, check_scalar, check_X_y, check_type, check_bound, \
    check_budget_manager
from skactiveml.utils import check_random_state, check_class_prior
from skactiveml.stream.budget_manager import SplitBudget


class TestValidation(unittest.TestCase):

    def test_check_scalar(self):
        x = 5
        self.assertRaises(TypeError, check_scalar, x=x, target_type=float,
                          name='x')
        self.assertRaises(ValueError, check_scalar, x=x, target_type=int,
                          max_val=4, name='x')
        self.assertRaises(ValueError, check_scalar, x=x, target_type=int,
                          max_inclusive=False, max_val=5, name='x')
        self.assertRaises(ValueError, check_scalar, x=x, target_type=int,
                          min_val=6, name='x')
        self.assertRaises(ValueError, check_scalar, x=x, target_type=int,
                          min_inclusive=False, min_val=5, name='x')

    def test_check_cost_matrix(self):
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[['2', '5'], ['a', '5']], n_classes=2)
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[[0, 1], [2, 0]], n_classes=3)
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[[0, 1], [2, 0]], n_classes=-1)
        self.assertRaises(TypeError, check_cost_matrix,
                          cost_matrix=[[0, 1], [2, 0]], n_classes=2.5)
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[[2, 1], [2, 2]], n_classes=2,
                          diagonal_is_zero=True)
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[[0, 1], [-1, 0]], n_classes=2,
                          only_non_negative=True)
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[[0, 0], [0, 0]], n_classes=2,
                          contains_non_zero=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_cost_matrix(cost_matrix=[[1, 1], [2, 0]], n_classes=2)
            check_cost_matrix(cost_matrix=[[0, 1], [-1, 0]], n_classes=2)
            check_cost_matrix(cost_matrix=[[0, 0], [0, 0]], n_classes=2)
            assert len(w) == 3

    def test_check_classes(self):
        self.assertRaises(TypeError, check_classes, classes=[None, 1, 2])
        self.assertRaises(TypeError, check_classes, classes=['2', 1, 2])
        self.assertRaises(TypeError, check_classes, classes=2)

    def test_check_class_prior(self):
        self.assertRaises(TypeError, check_class_prior, None, 1)
        self.assertRaises(TypeError, check_class_prior, 1, None)
        self.assertRaises(ValueError, check_class_prior, 1, 0)
        self.assertRaises(ValueError, check_class_prior, -2, 2)
        self.assertRaises(ValueError, check_class_prior, [0, 1, -1], 3)
        self.assertRaises(ValueError, check_class_prior, [1, 2, 3], 2)
        np.testing.assert_array_equal(check_class_prior(1, 3), [1, 1, 1])

    def test_check_missing_label(self):
        self.assertRaises(TypeError, check_missing_label, missing_label=[2])
        self.assertRaises(TypeError, check_missing_label, missing_label=self)
        self.assertRaises(TypeError, check_missing_label, missing_label=np.nan,
                          target_type=str)
        self.assertRaises(TypeError, check_missing_label, missing_label=2,
                          target_type=str)
        self.assertRaises(TypeError, check_missing_label, missing_label='2',
                          target_type=int)

    def test_check_X_y(self):
        X = [[1, 2], [3, 4]]
        y = [1, 0]
        X_cand = [[5, 6]]
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
            ValueError, check_X_y, X, y, X_cand_false, sample_weight,
            multi_output=True
        )
        y = np.array([[1, 0, 1], [2, 0, 1]])
        self.assertRaises(
            ValueError, check_X_y, X, y, X_cand, sample_weight,
            multi_output=True
        )

    def test_check_random_state(self):
        seed = 12
        self.assertRaises(ValueError, check_random_state, 'string')
        self.assertRaises(TypeError, check_random_state, seed, 'string')

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
        self.assertRaises(TypeError, check_type, 10, 'a', str)
        self.assertRaises(TypeError, check_type, 10, 'a', str, bool)
        self.assertRaises(TypeError, check_type, 10, 'a', str, bool, map, list)
        check_type(10, 'a', int)

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
        self.assertRaises(ValueError, check_bound, X=X,
                          bound_must_be_given=True)

    def test_check_budget_manager(self):

        self.assertIsNotNone(check_budget_manager(0.1, None, SplitBudget))
        with self.assertWarns(Warning):
            check_budget_manager(0.1, SplitBudget(budget=0.2), SplitBudget)
