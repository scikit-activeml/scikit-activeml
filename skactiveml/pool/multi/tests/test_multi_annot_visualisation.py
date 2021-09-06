import unittest

import numpy as np
import matplotlib.pyplot as plt

from skactiveml.classifier import PWC
from skactiveml.pool import UncertaintySampling, RandomSampler
from skactiveml.pool.multi._wrapper import MultiAnnotWrapper
from skactiveml.pool.multi.multi_annot_visualisation import plot_utility, \
    plot_data_set, plot_multi_annot_decision_boundary, check_or_get_bound


class TestMultiAnnotWrapper(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X = np.array([[0, 0], [0, 1], [1, 1]])
        self.y = np.array([[0, 1], [1, 1], [0, 0]])
        self.bound = np.array([[-0.5, -0.5], [1.5, 1.5]])

    def test_check_or_get_bound_X(self):
        self.assertRaises(ValueError, check_or_get_bound, X='string',
                          bound=self.bound)
        self.assertRaises(ValueError, check_or_get_bound,
                          X=np.array([-0.5, 1.5]), bound=self.bound)

    def test_check_or_get_bound_bound(self):
        self.assertRaises(ValueError, check_or_get_bound, X=self.X,
                          bound='string')
        self.assertRaises(ValueError, check_or_get_bound, X=self.X,
                          bound=np.array([0, 1]))
        self.assertWarns(Warning, check_or_get_bound, X=self.X,
                         bound=np.array([[0.5, 0.5], [1.5, 1.5]]))

    def ttest_plot_utility(self):
        clf = PWC()
        uncertainty = UncertaintySampling(clf=clf, method='entropy')
        wrapper = MultiAnnotWrapper(uncertainty, self.random_state,
                                    n_annotators=2)

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0, 0], [0, 1], [1, 1], [1, 1]])
        y_true = np.array([0, 0, 1, 1])
        bound = (-0.5, 1.5, -0.5, 1.5)

        clf.fit(X, y)
        fig = plot_utility(fig_size=(10, 10), ma_qs=wrapper, ma_qs_arg_dict={"X": X, "y": y},
                           bound=bound, res=5)

        plot_data_set(fig=fig, X=X, y=y, y_true=y_true)
        plot_multi_annot_decision_boundary(2, clf, fig=fig, bound=bound)

        plt.show()

    def ttext_plot_utility_us(self):
        qs = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(qs, self.random_state, n_annotators=2)

        plot_utility(fig_size=(3, 3), ma_qs=wrapper, ma_qs_arg_dict={},
                     bound=(0, 1, 0, 1), res=5)

        plt.show()

    def ttest_plot_utility_nearest_neighbour(self):
        qs = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(qs, self.random_state,
                                    n_annotators=2)

        X_cand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        plot_utility(fig_size=(3, 3), X_cand=X_cand, ma_qs=wrapper, ma_qs_arg_dict={}, bound=(0, 1, 0, 1),
                     res=5)

        plt.show()
