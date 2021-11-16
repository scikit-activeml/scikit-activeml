import unittest

import numpy as np

from skactiveml.utils import mesh, check_bound


class TestValidation(unittest.TestCase):

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

    def test_check_mesh(self):
        bound = np.array([[0, 0], [1, 1]])
        res = 10
        X_mesh, Y_mesh, mesh_instances = mesh(bound, res)

        self.assertEqual(X_mesh.shape, (10, 10))
        self.assertEqual(Y_mesh.shape, (10, 10))
        self.assertEqual(mesh_instances.shape, (100, 2))
