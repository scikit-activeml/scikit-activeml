import unittest
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import image_comparison
from sklearn.datasets import make_classification

from skactiveml.classifier import PWC
from skactiveml import visualization
from skactiveml.pool import UncertaintySampling
from skactiveml.visualization._feature_space import plot_decision_boundary, \
    plot_utility


class TestFeatureSpace(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.path.dirname(visualization.__file__) + \
                           '/tests/images/'
        np.random.seed(0)
        self.X, self.y = make_classification(n_features=2, n_redundant=0,
                                             random_state=0)
        train_indices = np.random.randint(0, len(self.X), size=20)
        cand_indices = np.setdiff1d(np.arange(len(self.X)), train_indices)
        self.X_train = self.X[train_indices]
        self.y_train = self.y[train_indices]
        self.X_cand = self.X[cand_indices]
        self.clf = PWC()
        self.qs = UncertaintySampling(clf=self.clf)
        self.qs_dict = {'X': self.X_train, 'y': self.y_train}

        x1_min = min(self.X[:, 0])
        x1_max = max(self.X[:, 0])
        x2_min = min(self.X[:, 1])
        x2_max = max(self.X[:, 1])
        self.bound = [[x1_min, x2_min], [x1_max, x2_max]]

        testing.set_font_settings_for_testing()
        testing.set_reproducibility_for_testing()
        testing.setup()

    def test_decision_boundary_clf(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.qs,
                          bound=self.bound)

    def test_decision_boundary_ax(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, ax=3)

    def test_utility_qs(self):
        self.assertRaises(TypeError, plot_utility, qs=self.clf,
                          qs_dict=self.qs_dict, bound=self.bound)

    def test_utility_qs_dict(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict={0, 1, 2}, bound=self.bound)

        qs_dict = self.qs_dict
        qs_dict['X_cand'] = []
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=qs_dict, bound=self.bound)

    def test_utility_X_cand(self):
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict)

    def test_utility_res(self):
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, bound=self.bound, res=-3)

    def test_utility_ax(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, bound=self.bound, ax=2)

    def test_no_candidates(self):
        self.clf.fit(self.X_train, self.y_train)
        bound = min(self.X[:, 0]), max(self.X[:, 0]), min(self.X[:, 1]), \
            max(self.X[:, 1])

        plot_utility(self.qs, {'X': self.X_train, 'y': self.y_train},
                     bound=bound)
        plt.scatter(self.X_cand[:, 0], self.X_cand[:, 1], c='k', marker='.')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=-self.y_train,
                    cmap='coolwarm_r', alpha=.9, marker='.')
        plot_decision_boundary(self.clf, bound)

        plt.savefig(self.path_prefix + 'test_result.png')
        comparison = compare_images(self.path_prefix +
                                    'visualization_without_candidates.png',
                                    self.path_prefix + 'test_result.png',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_with_candidates(self):
        self.clf.fit(self.X_train, self.y_train)
        bound = min(self.X[:, 0]), max(self.X[:, 0]), min(self.X[:, 1]), \
            max(self.X[:, 1])

        plot_utility(self.qs, {'X': self.X_train, 'y': self.y_train},
                     X_cand=self.X_cand, res=101)
        plt.scatter(self.X[:, 0], self.X[:, 1], c='k', marker='.')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=-self.y_train,
                    cmap='coolwarm_r', alpha=.9, marker='.')
        plot_decision_boundary(self.clf, bound)

        plt.savefig(self.path_prefix + 'test_result_cand.png')
        comparison = compare_images(self.path_prefix +
                                    'visualization_with_candidates.png',
                                    self.path_prefix + 'test_result_cand.png',
                                    tol=0)
        self.assertIsNone(comparison)