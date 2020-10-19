import numpy as np
import unittest

from skactiveml.pool import ALCE
from skactiveml.classifier import PWC


class TestPWC(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.X_cand = np.zeros((2, 2))
        self.classes = [0, 1, 2]
        self.C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.clf = PWC(classes=self.classes)

    def test_init(self):
        alce = ALCE(self.clf, self.C)
        alce.query(self.X_cand, self.X, self.y)

    def test_query(self):
        pass


if __name__ == '__main__':
    unittest.main()
