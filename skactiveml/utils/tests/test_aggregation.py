import numpy as np
import unittest

from skactiveml.utils import compute_vote_vectors, MISSING_LABEL


class TestAggregation(unittest.TestCase):

    def test_compute_vote_vectors(self):
        y = [['tokyo', 'paris', 'tokyo'], ['paris', 'paris', 'nan']]
        w = [[0.5, 1, 2], [0, 1, 0]]
        v_rec = compute_vote_vectors(y=y, w=w,
                                     classes=['tokyo', 'paris', 'new york'],
                                     missing_label='nan')
        v_exp = [[0, 1, 2.5], [0, 1, 0]]
        np.testing.assert_array_equal(v_rec, v_exp)

        y = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        w = [[0.5, 1, 2], [0, 1, 0]]
        v_rec = compute_vote_vectors(y=y, w=w, classes=[2, 4, 5],
                                     missing_label=np.nan)
        v_exp = [[0, 0, 0], [0, 0, 0]]
        np.testing.assert_array_equal(v_rec, v_exp)

    def test_compute_vote_vectors_all_nan(self):
        y = np.full(shape=(2, 3), fill_value=MISSING_LABEL)

        v_rec = compute_vote_vectors(y=y)
        v_exp = np.zeros((2, 1))

        np.testing.assert_array_equal(v_rec, v_exp)


if __name__ == '__main__':
    unittest.main()
