import numpy as np
import unittest

from ...utils import compute_vote_vectors


class TestAggregation(unittest.TestCase):

    def test_compute_vote_vectors(self):
        y = [['tokyo', 'paris', 'tokyo'], ['paris', 'paris', 'nan']]
        w = [[0.5, 1, 2], [0, 1, 0]]
        v_rec = compute_vote_vectors(y=y, w=w, classes=['tokyo', 'paris', 'new york'], unlabeled_class='nan')
        v_exp = [[0, 1, 2.5], [0, 1, 0]]
        np.testing.assert_array_equal(v_rec, v_exp)

        y = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        w = [[0.5, 1, 2], [0, 1, 0]]
        v_rec = compute_vote_vectors(y=y, w=w, classes=[2, 4, 5], unlabeled_class=np.nan)
        v_exp = [[0, 0, 0], [0, 0, 0]]
        np.testing.assert_array_equal(v_rec, v_exp)


if __name__ == '__main__':
    unittest.main()
