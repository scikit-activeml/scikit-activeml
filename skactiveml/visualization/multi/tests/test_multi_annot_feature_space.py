import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from sklearn.datasets import make_classification

from skactiveml import visualization
from skactiveml.classifier import PWC
from skactiveml.pool.multi import IEThresh
from skactiveml.utils import check_bound, majority_vote
from skactiveml.visualization.multi import plot_ma_data_set, plot_ma_utility, plot_ma_decision_boundary, \
    plot_ma_current_state


class TestFeatureSpace(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.path.dirname(visualization.__file__) + \
                           '/multi/tests/images/'

        self.X, self.y_true = make_classification(n_features=2, n_redundant=0,
                                                  random_state=0)

        self.n_samples = self.X.shape[0]
        self.n_annotators = 5

        rng = np.random.default_rng(seed=0)

        noise = rng.binomial(n=1, p=.2, size=(self.n_samples,
                                              self.n_annotators))

        self.y = (self.y_true.reshape(-1, 1) + noise) % 2

        self.clf = PWC(random_state=0)
        self.ma_qs = IEThresh(random_state=0, n_annotators=self.n_annotators)

        testing.set_font_settings_for_testing()
        testing.set_reproducibility_for_testing()
        testing.setup()

    def test_ma_plot_data_set_X(self):
        self.assertRaises(ValueError, plot_ma_data_set, self.X.T, self.y,
                          self.y_true)
        self.assertRaises(ValueError, plot_ma_data_set, self.X, self.y,
                          self.y_true.reshape(-1, 1))
        self.assertRaises(TypeError, plot_ma_data_set, self.X, self.y,
                          self.y_true, fig=4)

    def test_ma_plot_data_set(self):
        y = np.array(self.y, dtype=float)
        y[np.arange(5), np.arange(5)] = np.nan
        fig = plot_ma_data_set(self.X, y, self.y_true,
                               fig_size=(12, 3),
                               legend_dict={'loc': 'lower center',
                                            'bbox_to_anchor': (0.5, 0.1),
                                            'ncol': 3},
                               tick_dict={'labelbottom': True,
                                          'labelleft': True})

        fig.tight_layout()
        fig.savefig(self.path_prefix + 'data_set_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'data_set_expected_result.pdf',
                                    self.path_prefix +
                                    'data_set_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

        X_prime, y_true_prime = make_classification(n_features=2, n_redundant=0,
                                                    n_clusters_per_class=1,
                                                    n_classes=4, random_state=0)
        rng = np.random.default_rng(seed=0)

        noise = np.sum(rng.multinomial(n=1, pvals=[.7, .1, .1, .1],
                                       size=(self.n_samples,
                                             self.n_annotators)) \
                       * np.arange(4).reshape(1, 1, 4), axis=2)

        y_prime = (self.y_true.reshape(-1, 1) + noise) % 4

        fig = plot_ma_data_set(X_prime, y_prime, y_true_prime)
        fig.tight_layout()
        fig.savefig(self.path_prefix + 'data_set_mf_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'data_set_mf_expected_result.pdf',
                                    self.path_prefix +
                                    'data_set_mf_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_ma_plot_utility_args(self):
        y = np.array(self.y, dtype=float)
        y[np.arange(5), np.arange(5)] = np.nan
        maqs_arg_dict = {'clf': self.clf, 'X': self.X, 'y': self.y,
                         'X_cand': self.X}
        bound = check_bound(X=self.X)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          maqs_arg_dict, feature_bound=bound)
        maqs_arg_dict = {'clf': self.clf, 'X': self.X, 'y': self.y,
                         'A_cand': np.ones((self.n_samples, self.n_annotators))}
        bound = check_bound(X=self.X)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          maqs_arg_dict, feature_bound=bound)
        maqs_arg_dict = {'clf': self.clf, 'X': self.X, 'y': self.y}
        self.ma_qs.n_annotators = None
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          maqs_arg_dict, feature_bound=bound)
        fig, _ = plt.subplots(ncols=7)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          maqs_arg_dict, A_cand=np.ones((100, 5)), fig=fig,
                          feature_bound=bound)
        self.ma_qs.n_annotators = 5
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          maqs_arg_dict, fig=fig, feature_bound=bound)

    def test_ma_plot_utility(self):
        y = np.array(self.y, dtype=float)
        y[np.arange(5), np.arange(5)] = np.nan
        maqs_arg_dict = {'clf': self.clf, 'X': self.X, 'y': self.y}
        bound = check_bound(X=self.X)
        fig = plot_ma_utility(self.ma_qs, maqs_arg_dict, feature_bound=bound,
                              title='utility', fig_size=(20, 5))
        fig.tight_layout()
        fig.savefig(self.path_prefix + 'plot_utility_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'plot_utility_expected_result.pdf',
                                    self.path_prefix +
                                    'plot_utility_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

        maqs_arg_dict = {'clf': self.clf, 'X': self.X, 'y': self.y}
        A_cand = np.ones((self.n_samples, self.n_annotators))
        fig = plot_ma_utility(self.ma_qs, maqs_arg_dict, X_cand=self.X,
                              A_cand=A_cand)
        fig.tight_layout()
        fig.savefig(self.path_prefix + 'plot_utility_X_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'plot_utility_X_expected_result.pdf',
                                    self.path_prefix +
                                    'plot_utility_X_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_ma_plot_decision_boundary_args(self):
        bound = check_bound(X=self.X)
        self.assertRaises(ValueError, plot_ma_decision_boundary, self.clf,
                          bound)

    def test_ma_plot_decision_boundary(self):
        bound = check_bound(X=self.X)
        self.clf.fit(self.X, majority_vote(self.y, random_state=0))
        fig = plot_ma_decision_boundary(self.clf, bound,
                                        n_annotators=self.n_annotators)
        fig.tight_layout()
        fig.savefig(self.path_prefix +
                    'plot_decision_boundary_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'plot_decision_boundary'
                                    '_expected_result.pdf',
                                    self.path_prefix +
                                    'plot_decision_boundary'
                                    '_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_ma_plot_current_state(self):
        maqs_arg_dict = {'clf': self.clf, 'X': self.X, 'y': self.y}
        self.clf.fit(self.X, majority_vote(self.y, random_state=0))
        fig = plot_ma_current_state(self.X, self.y, self.y_true, self.ma_qs,
                                    self.clf, maqs_arg_dict)
        fig.tight_layout()
        fig.savefig(self.path_prefix +
                    'ma_plot_current_state_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'ma_plot_current_state_expected_result.pdf',
                                    self.path_prefix +
                                    'ma_plot_current_state_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)
