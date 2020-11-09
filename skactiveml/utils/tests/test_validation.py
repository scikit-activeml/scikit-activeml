import numpy as np
import unittest

from skactiveml.utils import check_cost_matrix, check_classes, \
    check_missing_label, check_scalar, check_X_y


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
                          cost_matrix=[[2, 1], [2, 2]], n_classes=3)
        self.assertRaises(ValueError, check_cost_matrix,
                          cost_matrix=[[2, 1], [2, 2]], n_classes=-1)
        self.assertRaises(TypeError, check_cost_matrix,
                          cost_matrix=[[2, 1], [2, 2]], n_classes=2.5)

    def test_check_classes(self):
        self.assertRaises(TypeError, check_classes, classes=[None, 1, 2])
        self.assertRaises(TypeError, check_classes, classes=['2', 1, 2])
        self.assertRaises(TypeError, check_classes, classes=2)

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
        self.assertRaises(ValueError, check_X_y, None, None)
        X = [[1, 2], [3, 4]]
        y = [1, 0]
        X_cand = [[5, 6]]
        sample_weight = [0.4, 0.6]
        check_X_y(X, y, X_cand, sample_weight)

if __name__ == '__main__':
    unittest.main()
