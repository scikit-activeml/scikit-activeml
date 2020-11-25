import unittest

import numpy as np

from skactiveml.pool import RandomSampler


class TestRandomSampler(unittest.TestCase):

    def setUp(self):
        self.X_cand = np.zeros((1000, 2))

    def test_init_param_random_state(self):
        rs = RandomSampler(random_state='string')
        self.assertTrue(hasattr(rs, 'random_state'))
        self.assertRaises(ValueError, rs.query, X_cand=[[1]])

    def test_query_param_X_cand(self):
        rand = RandomSampler()
        rand.query(self.X_cand)
        self.assertRaises(ValueError, rand.query, X_cand=[])

    def test_query(self):
        rand1 = RandomSampler(random_state=14)
        rand2 = RandomSampler(random_state=14)
        rand3 = RandomSampler(random_state=15)

        self.assertEqual(rand1.query(self.X_cand), rand2.query(self.X_cand))
        self.assertNotEqual(rand1.query(self.X_cand), rand3.query(self.X_cand))
        self.assertNotEqual(rand1.query(self.X_cand), rand2.query(self.X_cand))
