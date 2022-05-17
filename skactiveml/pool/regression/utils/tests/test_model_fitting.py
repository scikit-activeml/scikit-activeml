import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.pool.regression.utils import (
    update_X_y,
    update_reg,
    bootstrap_estimators,
)
from skactiveml.regressor import SklearnRegressor


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.reg = SklearnRegressor(GaussianProcessRegressor())
        self.X = np.arange(7 * 2).reshape(7, 2)
        self.y = np.arange(7)
        self.mapping = np.array([3, 4, 5])
        self.sample_weight = np.ones_like(self.y)
        self.x_pot = np.array([3, 4])
        self.y_pot = 5

    def test_update_X_y(self):

        X_new, y_new = update_X_y(
            self.X, self.y, self.y_pot, X_update=self.x_pot
        )

        self.assertEqual(X_new.shape, (8, 2))
        self.assertEqual(y_new.shape, (8,))
        np.testing.assert_equal(X_new[7], self.x_pot)
        self.assertEqual(y_new[7], self.y_pot)

        X_new, y_new = update_X_y(self.X, self.y, self.y_pot, idx_update=0)

        np.testing.assert_array_equal(X_new, self.X)
        self.assertEqual(y_new[0], 5)

        X_new, y_new = update_X_y(self.X, self.y, self.y, X_update=self.X)

        np.testing.assert_array_equal(X_new, np.append(self.X, self.X, axis=0))
        np.testing.assert_array_equal(y_new, np.append(self.y, self.y))

        X_new, y_new = update_X_y(
            self.X, self.y, np.array([3, 4]), idx_update=np.array([0, 2])
        )

        np.testing.assert_array_equal(X_new, self.X)
        self.assertEqual(y_new[0], 3)
        self.assertEqual(y_new[2], 4)

        self.assertRaises(ValueError, update_X_y, self.X, self.y, self.y_pot)

    def test_update_reg(self):
        self.assertRaises(
            (TypeError, ValueError),
            update_reg,
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            sample_weight=self.sample_weight,
            mapping=self.mapping,
        )
        self.reg.fit(self.X, self.y)
        reg_new = update_reg(
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            mapping=self.mapping,
            idx_update=1,
        )
        self.assertTrue(
            np.any(reg_new.predict(self.X) != self.reg.predict(self.X))
        )
        reg_new = update_reg(
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            mapping=self.mapping,
            idx_update=np.array([1]),
        )
        self.assertTrue(
            np.any(reg_new.predict(self.X) != self.reg.predict(self.X))
        )
        reg_new = update_reg(
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            mapping=None,
            X_update=np.array([8, 4]),
        )
        self.assertTrue(
            np.any(reg_new.predict(self.X) != self.reg.predict(self.X))
        )
        self.assertRaises(
            ValueError,
            update_reg,
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            sample_weight=np.arange(7) + 1,
            mapping=None,
            X_update=np.array([8, 4]),
        )

    def test_boostrap_aggregation(self):
        reg_s = bootstrap_estimators(self.reg, self.X, self.y, k_bootstrap=5)
        self.assertEqual(len(reg_s), 5)

        reg_s = bootstrap_estimators(
            self.reg,
            self.X,
            self.y,
            sample_weight=self.sample_weight,
            k_bootstrap=5,
        )
        self.assertEqual(len(reg_s), 5)
