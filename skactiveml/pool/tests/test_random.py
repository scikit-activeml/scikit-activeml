import unittest

import numpy as np

from skactiveml.pool import RandomSampler


class TestEER(unittest.TestCase):

    def setUp(self):
        self.X_cand = np.zeros((1000, 2))

    def test_init(self):
        self.assertRaises(ValueError, RandomSampler, random_state='string')

    def test_query(self):
        rand = RandomSampler()
        rand.query(self.X_cand)
        self.assertRaises(ValueError, rand.query, X_cand=[])

        rand1 = RandomSampler(random_state=14)
        rand2 = RandomSampler(random_state=14)
        rand3 = RandomSampler(random_state=15)

        self.assertEqual(rand1.query(self.X_cand), rand2.query(self.X_cand))
        self.assertNotEqual(rand1.query(self.X_cand), rand3.query(self.X_cand))
        self.assertNotEqual(rand1.query(self.X_cand), rand2.query(self.X_cand))


if __name__ == '__main__':
    unittest.main()
