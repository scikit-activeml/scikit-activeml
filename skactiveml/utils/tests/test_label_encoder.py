import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from skactiveml.utils import ExtLabelEncoder


class TestLabelEncoder(unittest.TestCase):
    def setUp(self):
        self.y1 = [np.nan, 2, 5, 10, np.nan]
        self.y2 = [np.nan, "2", "5", "10", np.nan]
        self.y3 = [None, 2, 5, 10, None]
        self.y4 = [None, "2", "5", "10", None]
        self.y5 = [8, -1, 1, 5, 2]
        self.y6 = ["paris", "france", "tokyo", "nan"]
        self.y7 = ["paris", "france", "tokyo", -1]

    def test_ExtLabelEncoder(self):
        ext_le = ExtLabelEncoder(classes=[2, "2"])
        self.assertRaises(TypeError, ext_le.fit, self.y1)
        ext_le = ExtLabelEncoder(classes=["1", "2"], missing_label=np.nan)
        self.assertRaises(TypeError, ext_le.fit, self.y1)
        self.assertRaises(
            NotFittedError, ExtLabelEncoder().transform, y=["1", "2"]
        )

        # missing_label=np.nan
        ext_le = ExtLabelEncoder().fit(self.y1)
        y_enc = ext_le.transform(self.y1)
        np.testing.assert_array_equal([-1, 0, 1, 2, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y1, y_dec)

        # missing_label=None
        ext_le = ExtLabelEncoder(missing_label=None).fit(self.y3)
        y_enc = ext_le.transform(self.y3)
        np.testing.assert_array_equal([-1, 0, 1, 2, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y3, y_dec)
        ext_le = ExtLabelEncoder(missing_label=None).fit(self.y4)
        y_enc = ext_le.transform(self.y4)
        np.testing.assert_array_equal([-1, 1, 2, 0, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y4, y_dec)

        # missing_label=-1
        ext_le = ExtLabelEncoder(missing_label=-1).fit(self.y5)
        y_enc = ext_le.transform(self.y5)
        np.testing.assert_array_equal([3, -1, 0, 2, 1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y5, y_dec)

        # missing_label='nan'
        ext_le = ExtLabelEncoder(missing_label="nan").fit(self.y6)
        y_enc = ext_le.transform(self.y6)
        np.testing.assert_array_equal([1, 0, 2, -1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y6, y_dec)

        # classes=[0, 2, 5, 10], missing_label=np.nan
        cls = [0, 2, 5, 10]
        np.testing.assert_array_equal(
            [-1, 1, 2, 3, -1],
            ExtLabelEncoder(classes=cls).fit_transform(self.y1),
        )
