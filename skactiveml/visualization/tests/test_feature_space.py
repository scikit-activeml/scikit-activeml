import unittest
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

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
        self.clf.fit(self.X_train, self.y_train)
        self.qs = UncertaintySampling()
        self.qs_dict = {'clf': self.clf, 'X': self.X_train, 'y': self.y_train}

        x1_min = min(self.X[:, 0])
        x1_max = max(self.X[:, 0])
        x2_min = min(self.X[:, 1])
        x2_max = max(self.X[:, 1])
        self.bound = [[x1_min, x2_min], [x1_max, x2_max]]

        self.cmap = 'jet'

        testing.set_font_settings_for_testing()
        testing.set_reproducibility_for_testing()
        testing.setup()

    # Tests for plot_decision_boundary function
    def test_decision_boundary_clf(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.qs,
                          feature_bound=self.bound)
        clf = TestClassifier()
        self.assertRaises(AttributeError, plot_decision_boundary, clf=clf,
                          feature_bound=self.bound)

    def test_decision_boundary_bound(self):
        self.assertRaises(ValueError, plot_decision_boundary, clf=self.clf,
                          feature_bound=[0, 0, 1, 1])

    def test_decision_boundary_res(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, res='string')

    def test_decision_boundary_ax(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, ax=3)

    def test_decision_boundary_confidence(self):
        self.assertRaises(ValueError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, confidence=0.0)
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, confidence='string')
        plot_decision_boundary(self.clf, self.bound, confidence=None)
        svc = LinearSVC()
        svc.fit(self.X_train, self.y_train)
        self.assertWarns(Warning, plot_decision_boundary, clf=svc,
                         feature_bound=self.bound, confidence=0.75)

    def test_decision_boundary_cmap(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, cmap=4)

    def test_decision_boundary_boundary_dict(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, boundary_dict='string')
        plot_decision_boundary(clf=self.clf, feature_bound=self.bound,
                               boundary_dict={'colors': 'r'})

    def test_decision_boundary_confidence_dict(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          feature_bound=self.bound, confidence_dict='string')
        plot_decision_boundary(clf=self.clf, feature_bound=self.bound,
                               confidence_dict={'linestyles': ':'})

    # Tests for plot_utility function
    def test_utility_qs(self):
        self.assertRaises(TypeError, plot_utility, qs=self.clf,
                          qs_dict=self.qs_dict, feature_bound=self.bound)

    def test_utility_qs_dict(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict={0, 1, 2}, feature_bound=self.bound)

        qs_dict = self.qs_dict
        qs_dict['X_cand'] = []
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=qs_dict, feature_bound=self.bound)

    def test_utility_X_cand(self):
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict)

    def test_utility_res(self):
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, feature_bound=self.bound,
                          res=-3)

    def test_utility_ax(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, feature_bound=self.bound, ax=2)

    def test_utility_contour_dict(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, feature_bound=self.bound,
                          contour_dict='string')
        plot_utility(qs=self.qs, qs_dict=self.qs_dict,
                     feature_bound=self.bound,
                     contour_dict={'linestyles': '.'})

    # Graphical tests
    def test_no_candidates(self):
        fig, ax = plt.subplots()
        plot_utility(self.qs, self.qs_dict, feature_bound=self.bound, ax=ax)
        ax.scatter(self.X_cand[:, 0], self.X_cand[:, 1], c='k', marker='.')
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train,
                   cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(self.clf, self.bound, ax=ax, cmap=self.cmap)

        fig.savefig(self.path_prefix + 'dec_bound_wo_cand.pdf')
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_wo_cand_base.pdf',
                                    self.path_prefix + 'dec_bound_wo_cand.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_with_candidates(self):
        fig, ax = plt.subplots()
        plot_utility(self.qs, self.qs_dict, X_cand=self.X_cand, ax=ax)
        ax.scatter(self.X[:, 0], self.X[:, 1], c='k', marker='.')
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train,
                   cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(self.clf, self.bound, ax=ax, cmap=self.cmap)

        fig.savefig(self.path_prefix + 'dec_bound_w_cand.pdf')
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_w_cand_base.pdf',
                                    self.path_prefix + 'dec_bound_w_cand.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_multi_class(self):
        X, y = make_classification(n_features=2, n_redundant=0, random_state=0,
                                   n_classes=3, n_clusters_per_class=1)
        train_indices = np.random.randint(0, len(X), size=20)
        cand_indices = np.setdiff1d(np.arange(len(X)), train_indices)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_cand = X[cand_indices]
        clf = PWC()
        clf.fit(X_train, y_train)
        qs = UncertaintySampling()
        bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]

        fig, ax = plt.subplots()
        plot_utility(qs, {'clf': clf, 'X': X_train, 'y': y_train},
                     feature_bound=bound, ax=ax)
        ax.scatter(X_cand[:, 0], X_cand[:, 1], c='k', marker='.')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                   cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(clf, bound, ax=ax, res=101, cmap=self.cmap)
        fig.savefig(self.path_prefix + 'dec_bound_multiclass.pdf')
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_multiclass_base.pdf',
                                    self.path_prefix +
                                    'dec_bound_multiclass.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_svc(self):
        svc = LinearSVC()
        svc.fit(self.X_train, self.y_train)

        fig, ax = plt.subplots()
        plot_utility(self.qs, self.qs_dict, X_cand=self.X_cand, ax=ax)
        ax.scatter(self.X[:, 0], self.X[:, 1], c='k', marker='.')
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train,
                   cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(svc, self.bound, ax=ax, cmap=self.cmap)

        fig.savefig(self.path_prefix + 'dec_bound_svc.pdf')
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_svc_base.pdf',
                                    self.path_prefix + 'dec_bound_svc.pdf',
                                    tol=0)
        self.assertIsNone(comparison)


class TestClassifier(ClassifierMixin):
    pass
