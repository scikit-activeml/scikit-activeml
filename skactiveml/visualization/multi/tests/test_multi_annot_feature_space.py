import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from sklearn.datasets import make_classification

from skactiveml import visualization
from skactiveml.visualization.multi import plot_data_set


class TestFeatureSpace(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.path.dirname(visualization.__file__) + \
                           '/multi/tests/images/'
        np.random.seed(0)
        self.X, self.y_true = make_classification(n_features=2, n_redundant=0,
                                                  random_state=0)

        self.n_samples = self.X.shape[0]
        self.n_annotators = 5

        rng = np.random.default_rng(seed=0)

        noise = rng.binomial(n=1, p=.2, size=(self.n_samples,
                                              self.n_annotators))

        self.y = (self.y_true.reshape(-1, 1) + noise) % 2

    def test_plot_data_set_X(self):
        self.assertRaises(ValueError, plot_data_set, self.X.flatten(), self.y,
                          self.y_true)
        self.assertRaises(ValueError, plot_data_set, self.X, self.y,
                          self.y_true.reshape(-1, 1))

    def test_plot_data_set(self):
        y = np.array(self.y, dtype=float)
        y[np.arange(5), np.arange(5)] = np.nan
        fig = plot_data_set(self.X, y, self.y_true,
                            legend_dict={'loc': 'lower center'})
        fig.tight_layout()
        #fig.savefig(self.path_prefix + 'data_set_returned_result.pdf')
        #comparison = compare_images(self.path_prefix +
        #                            'data_set_expected_result.pdf',
        #                            self.path_prefix +
        #                            'data_set_returned_result.pdf',
        #                            tol=0)
        #self.assertIsNone(comparison)
        plt.show()
