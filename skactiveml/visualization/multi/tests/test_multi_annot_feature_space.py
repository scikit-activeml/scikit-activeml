import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from sklearn.datasets import make_classification

from skactiveml import visualization
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.classifier.multiannotator import AnnotatorEnsembleClassifier
from skactiveml.pool.multiannotator import IntervalEstimationThreshold
from skactiveml.utils import check_bound, majority_vote
from skactiveml.visualization.multi import plot_ma_data_set, plot_ma_utility, \
    plot_ma_decision_boundary, \
    plot_ma_current_state


class TestFeatureSpace(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.path.dirname(visualization.__file__) + \
                           '/multiannotator/tests/images/'

        self.X, self.y_true = make_classification(n_features=2, n_redundant=0,
                                                  random_state=0)

        self.n_samples = self.X.shape[0]
        self.n_annotators = 5

        rng = np.random.default_rng(seed=0)

        noise = rng.binomial(n=1, p=.2, size=(self.n_samples,
                                              self.n_annotators))

        self.y = (self.y_true.reshape(-1, 1) + noise) % 2

        estimators = []
        for a in range(self.n_annotators):
            estimators.append((f'pwc_{a}', ParzenWindowClassifier(random_state=0)))
        self.clf_multi = AnnotatorEnsembleClassifier(
            estimators=estimators, voting='soft'
        )
        self.clf = ParzenWindowClassifier(random_state=0)
        self.ma_qs = IntervalEstimationThreshold(random_state=0)

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
                                    tol=0.0)
        self.assertIsNone(comparison)

    def test_ma_plot_data_set_mc(self):
        X_prime, y_true_prime = make_classification(n_features=2,
                                                    n_redundant=0,
                                                    n_clusters_per_class=1,
                                                    n_classes=4,
                                                    random_state=0)
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
        ma_qs_arg_dict = {'clf': self.clf}
        bound = check_bound(X=self.X)
        self.assertRaises(ValueError, plot_ma_utility, ma_qs=self.ma_qs,
                          X=self.X, y=self.y, ma_qs_arg_dict=ma_qs_arg_dict,
                          feature_bound=bound)
        bound = check_bound(X=self.X)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          X=self.X, y=self.y, ma_qs_arg_dict=ma_qs_arg_dict,
                          feature_bound=bound)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          X=self.X, y=self.y, ma_qs_arg_dict=ma_qs_arg_dict,
                          feature_bound=bound)
        fig, _ = plt.subplots(ncols=7)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          X=self.X, y=self.y, ma_qs_arg_dict=ma_qs_arg_dict,
                          annotators=np.ones((100, 5)), fig=fig,
                          feature_bound=bound)
        self.assertRaises(ValueError, plot_ma_utility, self.ma_qs,
                          ma_qs_arg_dict={'clf': self.clf, 'candidates':None},
                          feature_bound=bound, X=self.X, y=self.y,
                          title='utility', fig_size=(20, 5))

    def test_ma_plot_utility(self):
        y = np.array(self.y, dtype=float)
        y[np.arange(5), np.arange(5)] = np.nan
        ma_qs_arg_dict = {'clf': self.clf_multi}
        bound = check_bound(X=self.X)
        fig = plot_ma_utility(self.ma_qs, ma_qs_arg_dict=ma_qs_arg_dict,
                              feature_bound=bound, X=self.X, y=self.y,
                              title='utility', fig_size=(20, 5))
        fig.tight_layout()
        fig.savefig(self.path_prefix + 'plot_utility_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'plot_utility_expected_result.pdf',
                                    self.path_prefix +
                                    'plot_utility_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)
        ma_qs = IntervalEstimationThreshold(random_state=0)
        fig = plot_ma_utility(ma_qs, ma_qs_arg_dict=ma_qs_arg_dict,
                              feature_bound=bound, X=self.X, y=self.y,
                              title='utility', fig_size=(20, 5),
                              candidates=np.arange(10))
        fig.tight_layout()
        fig.savefig(self.path_prefix + 'plot_utility_2_returned_result.pdf')
        comparison = compare_images(self.path_prefix +
                                    'plot_utility_2_expected_result.pdf',
                                    self.path_prefix +
                                    'plot_utility_2_returned_result.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_ma_plot_utility_with_X(self):
        maqs_arg_dict = {'clf': self.clf_multi}
        fig = plot_ma_utility(self.ma_qs, ma_qs_arg_dict=maqs_arg_dict,
                              X=self.X, y=self.y,
                              candidates=self.X)
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
        maqs_arg_dict = {'clf': self.clf, 'fit_clf': False}
        self.clf.fit(self.X, majority_vote(self.y, random_state=0))
        fig = plot_ma_current_state(self.X, self.y, self.y_true, self.ma_qs,
                                    self.clf, maqs_arg_dict)
        fig.tight_layout()
        fig.savefig(self.path_prefix +
                    'ma_plot_current_state_returned_result.pdf')
        comparison = compare_images(
            self.path_prefix + 'ma_plot_current_state_expected_result.pdf',
            self.path_prefix + 'ma_plot_current_state_returned_result.pdf',
            tol=0
        )
        self.assertIsNone(comparison)
