import numpy as np
import unittest

from skactiveml.utils import check_cost_matrix, check_classes, \
    check_missing_label


class TestValidation(unittest.TestCase):

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
        np.testing.assert_array_equal([2, 3, 1], check_classes([2, 3, 1]))
        np.testing.assert_array_equal(['2', '3', '1'],
                                      check_classes(['2', '3', '1']))

    def test_check_missing_label(self):
        self.assertRaises(TypeError, check_missing_label, missing_label=[2])
        self.assertRaises(TypeError, check_missing_label, missing_label=self)
        self.assertRaises(TypeError, check_missing_label, missing_label=np.nan,
                          target_type=str)
        self.assertRaises(TypeError, check_missing_label, missing_label=2,
                          target_type=str)
        self.assertRaises(TypeError, check_missing_label, missing_label='2',
                          target_type=int)
        self.assertIs(np.nan, check_missing_label(missing_label=np.nan,
                                                  target_type=int))
        self.assertIs(None,
                      check_missing_label(missing_label=None, target_type=int))
        self.assertEqual('2', check_missing_label(missing_label='2',
                                                  target_type=str))
        self.assertEqual(2.5, check_missing_label(missing_label=2.5,
                                                  target_type=int))


if __name__ == '__main__':
    unittest.main()
