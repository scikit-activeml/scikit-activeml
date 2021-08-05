import unittest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling, RandomSampler
from skactiveml.pool.multi._wrapper import MultiAnnotWrapper
from skactiveml.pool.multi.multi_annot_visualisation import plot_scores_2d, plot_utility, plot_utility_difference


class TestMultiAnnotWrapper(unittest.TestCase):

    def setUp(self):
        self.random_state = 1

    def test_plot_scores_2d(self):
        X = np.array([[0, 0], [0, 1], [1, 1]])

        y = np.array([[0, 1], [1, 1], [0, 0]])

        y_true = np.array([0, 1, 1])

        plot_scores_2d(figsize=(3, 4), X=X, y=y, y_true=y_true)
        plt.show()

    def test_plot_utility(self):
        qs = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(qs, self.random_state,
                                    n_annotators=2)

        plot_utility(figsize=(3, 3), qs=wrapper, qs_dict={}, bound=(0, 1, 0, 1),
                     res=5)

        plt.show()

    def test_plot_utility_nearest_neighbour(self):
        qs = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(qs, self.random_state,
                                    n_annotators=2)

        X_cand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        plot_utility(figsize=(3, 3), X_cand=X_cand, qs=wrapper, qs_dict={}, bound=(0, 1, 0, 1),
                     res=5)

        plt.show()

    def test_plot_utility_difference(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)

        uncertainty = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(uncertainty, self.random_state,
                                    n_annotators=2)

        plot_utility_difference(figsize=(3, 3), qs=wrapper, qs_dict={}, bound=(0, 1, 0, 1),
                     res=5)

        plt.show()
