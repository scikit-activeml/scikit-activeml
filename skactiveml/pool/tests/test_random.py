import unittest

import numpy as np
from numpy.random import RandomState

from skactiveml.pool import RandomSampler
from skactiveml.utils import MISSING_LABEL


class TestRandomSampler(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((10, 2))
        self.y = np.full(10, MISSING_LABEL)

    def test_query(self):
        rand1 = RandomSampler(random_state=RandomState(14))
        rand2 = RandomSampler(random_state=14)

        self.assertEqual(rand1.query(self.X, self.y),
                         rand1.query(self.X, self.y))
        self.assertEqual(rand1.query(self.X, self.y),
                         rand2.query(self.X, self.y))

        qidx = rand1.query(self.X, self.y)
        self.assertEqual(len(qidx), 1)

