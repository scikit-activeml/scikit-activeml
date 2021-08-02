import unittest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.pool.multi._wrapper import MultiAnnotWrapper
from skactiveml.pool.multi.multi_annot_visualisation import plot_scores_2d


class TestMultiAnnotWrapper(unittest.TestCase):

    def setUp(self):
        self.random_state = 1

    def testPlotting(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)

        uncertainty = UncertaintySampling(clf=clf, method='entropy')

        wrapper = MultiAnnotWrapper(uncertainty, self.random_state)

        X = np.array([[0, 0], [0, 1], [1, 1]])

        y = np.array([[0, 1], [1, 1], [0, 0]])

        y_true = np.array([0, 1, 1])

        plot_scores_2d(figsize=(3, 3), X=X, y=y, y_true=y_true)
        plt.show()
