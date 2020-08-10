import numpy as np
import unittest

from ...utils import ExtLabelEncoder


class TestLabel(unittest.TestCase):

    def setUp(self):
        self.y1 = [np.nan, 2, 5, 10, np.nan]
        self.y2 = [np.nan, '2', '5', '10', np.nan]
        self.y3 = [None, 2, 5, 10, None]
        self.y4 = [None, '2', '5', '10', None]
        self.y5 = [8, -1, 1, 5, 2]
        self.y6 = ['paris', 'france', 'tokyo', 'nan']
        self.y7 = ['paris', 'france', 'tokyo', -1]

    def test_ExtLabelEncoder(self):
        # unlabeled_class=np.nan
        ext_le = ExtLabelEncoder().fit(self.y1)
        y_enc = ext_le.transform(self.y1)
        np.testing.assert_array_equal([np.nan, 0, 1, 2, np.nan], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y1, y_dec)
        ext_le = ExtLabelEncoder().fit(self.y2)
        y_enc = ext_le.transform(self.y2)
        np.testing.assert_array_equal([np.nan, 1, 2, 0, np.nan], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y2, y_dec)

        # unlabeled_class=None
        ext_le = ExtLabelEncoder(unlabeled_class=None).fit(self.y3)
        y_enc = ext_le.transform(self.y3)
        np.testing.assert_array_equal([np.nan, 0, 1, 2, np.nan], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y3, y_dec)
        ext_le = ExtLabelEncoder(unlabeled_class=None).fit(self.y4)
        y_enc = ext_le.transform(self.y4)
        np.testing.assert_array_equal([np.nan, 1, 2, 0, np.nan], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y4, y_dec)

        # unlabeled_class=-1
        ext_le = ExtLabelEncoder(unlabeled_class=-1).fit(self.y5)
        y_enc = ext_le.transform(self.y5)
        np.testing.assert_array_equal([3, np.nan, 0, 2, 1], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y5, y_dec)

        # unlabeled_class='nan'
        ext_le = ExtLabelEncoder(unlabeled_class='nan').fit(self.y6)
        y_enc = ext_le.transform(self.y6)
        np.testing.assert_array_equal([1, 0, 2, np.nan], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y6, y_dec)

        # unlabeled_class=-1
        ext_le = ExtLabelEncoder(unlabeled_class='-1').fit(self.y7)
        y_enc = ext_le.transform(self.y7)
        np.testing.assert_array_equal([1, 0, 2, np.nan], y_enc)
        y_dec = ext_le.inverse_transform(y_enc)
        np.testing.assert_array_equal(self.y7, y_dec)

        # classes=[0,1,2], unlabeled_class=np.nan
        cls = [0, 2, 5, 10]
        np.testing.assert_array_equal([np.nan, 1, 2, 3, np.nan], ExtLabelEncoder(classes=cls).fit_transform(self.y1))


if __name__ == '__main__':
    unittest.main()
