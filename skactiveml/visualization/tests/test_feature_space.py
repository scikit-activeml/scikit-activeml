import os
import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from skactiveml import visualization
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import (
    UncertaintySampling,
    RandomSampling,
    ValueOfInformationEER,
)
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.visualization import (
    plot_decision_boundary,
    plot_utilities,
    plot_contour_for_samples,
    plot_annotator_utilities,
)
from skactiveml.visualization._feature_space import _general_plot_utilities


class TestFeatureSpace(unittest.TestCase):
    def setUp(self):
        self.path_prefix = (
                os.path.dirname(visualization.__file__) + "/tests/images/"
        )
        self.X, self.y = make_classification(
            n_features=2, n_redundant=0, random_state=0
        )
        train_indices = np.random.RandomState(0).randint(
            0, len(self.X), size=20
        )
        cand_indices = np.setdiff1d(np.arange(len(self.X)), train_indices)
        self.y_active = np.full_like(self.y, np.nan, dtype=float)
        self.y_active[cand_indices] = self.y[cand_indices]
        self.y_active_multi = np.tile(self.y_active, [5, 1]).T
        self.X_train = self.X[train_indices]
        self.y_train = self.y[train_indices]
        self.y_train_multi = np.tile(self.y_train, [5, 1]).T
        self.X_cand = self.X[cand_indices]
        self.clf = ParzenWindowClassifier(random_state=0)
        self.clf.fit(self.X_train, self.y_train)
        self.qs = UncertaintySampling(random_state=0)
        self.qs_dict = {"clf": self.clf}
        self.utilities = clone(self.qs).query(
            X=self.X,
            y=self.y,
            clf=self.clf,
            candidates=self.X,
            return_utilities=True,
        )[1][0]

        x1_min = min(self.X[:, 0])
        x1_max = max(self.X[:, 0])
        x2_min = min(self.X[:, 1])
        x2_max = max(self.X[:, 1])
        self.bound = [[x1_min, x2_min], [x1_max, x2_max]]

        self.cmap = "jet"

        testing.set_font_settings_for_testing()
        testing.set_reproducibility_for_testing()
        testing.setup()

    # Tests for plot_decision_boundary function
    def test_decision_boundary_param_clf(self):
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.qs,
            feature_bound=self.bound,
        )
        clf = TestClassifier()
        self.assertRaises(
            AttributeError,
            plot_decision_boundary,
            clf=clf,
            feature_bound=self.bound,
        )

    def test_decision_boundary_param_bound(self):
        self.assertRaises(
            ValueError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=[0, 0, 1, 1],
        )

    def test_decision_boundary_param_res(self):
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            res="string",
        )

    def test_decision_boundary_param_ax(self):
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            ax=3,
        )

    def test_decision_boundary_param_confidence(self):
        self.assertRaises(
            ValueError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            confidence=0.0,
        )
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            confidence="string",
        )
        plot_decision_boundary(self.clf, self.bound, confidence=None)
        svc = LinearSVC()
        svc.fit(self.X_train, self.y_train)
        self.assertWarns(
            Warning,
            plot_decision_boundary,
            clf=svc,
            feature_bound=self.bound,
            confidence=0.75,
        )

    def test_decision_boundary_param_cmap(self):
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            cmap=4,
        )

    def test_decision_boundary_param_boundary_dict(self):
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            boundary_dict="string",
        )
        plot_decision_boundary(
            clf=self.clf,
            feature_bound=self.bound,
            boundary_dict={"colors": "r"},
        )

    def test_decision_boundary_param_confidence_dict(self):
        self.assertRaises(
            TypeError,
            plot_decision_boundary,
            clf=self.clf,
            feature_bound=self.bound,
            confidence_dict="string",
        )
        plot_decision_boundary(
            clf=self.clf,
            feature_bound=self.bound,
            confidence_dict={"linestyles": ":"},
        )

    # Tests for plot_utilities function
    def test__general_plot_utilities_param_qs(self):
        self.assertRaises(
            TypeError,
            _general_plot_utilities,
            qs=self.clf,
            X=self.X,
            y=self.y,
            **self.qs_dict,
            feature_bound=self.bound
        )

    def test__general_plot_utilities_param_X(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=np.ones([len(self.X), 3]),
            y=self.y,
            **self.qs_dict,
            feature_bound=self.bound
        )

    def test__general_plot_utilities_param_y(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=np.zeros(len(self.y) + 1),
            **self.qs_dict,
            feature_bound=self.bound
        )

    def test__general_plot_utilities_param_candidates(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y,
            **self.qs_dict,
            candidates=[100]
        )
        _general_plot_utilities(
            qs=self.qs, X=self.X, y=self.y, **self.qs_dict, candidates=[99]
        )

    def test__general_plot_utilities_param_replace_nan(self):
        _general_plot_utilities(
            qs=self.qs,
            X=self.X,
            y=self.y,
            candidates=[1],
            **self.qs_dict,
            replace_nan=None,
            feature_bound=self.bound
        )

    def test__general_plot_utilities_param_plot_annotators(self):
        self.assertRaises(
            TypeError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y,
            **self.qs_dict,
            plot_annotators=[4]
        )
        _, axes = plt.subplots(1, 2)
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y_active_multi,
            **self.qs_dict,
            plot_annotators=[4],
            axes=axes
        )

    def test__general_plot_utilities_param_ignore_undefined_query_params(self):
        _general_plot_utilities(
            qs=ValueOfInformationEER(),
            X=self.X,
            y=self.y_active,
            **self.qs_dict,
            ignore_undefined_query_params=True,
            feature_bound=self.bound
        )
        _general_plot_utilities(
            qs=self.qs,
            X=self.X,
            y=self.y,
            candidates=None,
            **self.qs_dict,
            ignore_undefined_query_params=True,
            feature_bound=self.bound
        )
        _general_plot_utilities(
            qs=self.qs,
            X=self.X,
            y=self.y,
            candidates=[1],
            **self.qs_dict,
            ignore_undefined_query_params=True,
            feature_bound=self.bound
        )

    def test__general_plot_utilities_param_res(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y_active,
            **self.qs_dict,
            feature_bound=self.bound,
            res=-3
        )

    def test__general_plot_utilities_param_ax(self):
        self.assertRaises(
            TypeError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y_active,
            **self.qs_dict,
            feature_bound=self.bound,
            ax=2
        )
        _, axes = plt.subplots(1, 2)
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y_active_multi,
            **self.qs_dict,
            feature_bound=self.bound,
            ax=axes
        )

    def test__general_plot_utilities_param_axes(self):
        self.assertRaises(
            TypeError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y_active,
            **self.qs_dict,
            feature_bound=self.bound,
            axes=2
        )

    def test__general_plot_utilities_param_contour_dict(self):
        self.assertRaises(
            TypeError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y_active,
            **self.qs_dict,
            feature_bound=self.bound,
            contour_dict="string"
        )
        _general_plot_utilities(
            qs=self.qs,
            **self.qs_dict,
            X=self.X,
            y=self.y,
            feature_bound=self.bound,
            contour_dict={"linestyles": "."}
        )

    def test_plot_contour_for_samples_param_X(self):
        for X in [None, 1, np.arange(10)]:
            self.assertRaises(
                ValueError,
                plot_contour_for_samples,
                X=X,
                values=self.utilities,
            )

    def test_plot_contour_for_samples_param_values(self):
        test_cases = [
            (None, TypeError),
            (1, TypeError),
            (np.arange(10), ValueError),
        ]
        for values, err in test_cases:
            self.assertRaises(
                err, plot_contour_for_samples, X=self.X, values=values
            )

    def test_plot_contour_for_samples_param_replace_nan(self):
        values = np.full_like(self.utilities, np.nan)
        for nan, err in [(np.nan, ValueError), ("s", TypeError)]:
            self.assertRaises(
                err,
                plot_contour_for_samples,
                X=self.X,
                values=values,
                replace_nan=nan,
            )

    def test_plot_contour_for_samples_param_feature_bound(self):
        test_cases = [
            (np.nan, ValueError),
            ("s", ValueError),
            ((2, 1), ValueError),
        ]
        for b, err in test_cases:
            self.assertRaises(
                err,
                plot_contour_for_samples,
                X=self.X,
                values=self.utilities,
                feature_bound=b,
            )

    def test_plot_contour_for_samples_param_ax(self):
        test_cases = [
            (np.nan, AttributeError),
            ("s", AttributeError),
            ((2, 1), AttributeError),
        ]
        for ax, err in test_cases:
            self.assertRaises(
                err,
                plot_contour_for_samples,
                X=self.X,
                values=self.utilities,
                ax=ax,
            )

    def test_plot_contour_for_samples_param_res(self):
        test_cases = [
            (np.nan, TypeError),
            ("s", TypeError),
            ((2, 1), TypeError),
            (-1, ValueError),
        ]
        for res, err in test_cases:
            self.assertRaises(
                err,
                plot_contour_for_samples,
                X=self.X,
                values=self.utilities,
                res=res,
            )

    def test_plot_contour_for_samples_param_contour_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("s", TypeError),
            ((2, 1), TypeError),
            (-1, TypeError),
        ]
        for cont, err in test_cases:
            self.assertRaises(
                err,
                plot_contour_for_samples,
                X=self.X,
                values=self.utilities,
                contour_dict=cont,
            )
        plot_contour_for_samples(
            X=self.X, values=self.utilities, contour_dict={"linestyles": "."}
        )

    # Graphical tests

    def test_without_candidates(self):
        fig, ax = plt.subplots()
        qs = RandomSampling(random_state=0)
        plot_utilities(
            qs=qs,
            X=np.zeros((1, 2)),
            y=[np.nan],
            feature_bound=self.bound,
            ax=ax,
        )

        ax.scatter(self.X_cand[:, 0], self.X_cand[:, 1], c="k", marker=".")
        ax.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=self.y_train,
            cmap=self.cmap,
            alpha=0.9,
            marker=".",
        )
        plot_decision_boundary(self.clf, self.bound, ax=ax, cmap=self.cmap)

        fig.savefig(self.path_prefix + "dec_bound_wo_cand.pdf")
        comparison = compare_images(
            self.path_prefix + "dec_bound_wo_cand_expected.pdf",
            self.path_prefix + "dec_bound_wo_cand.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)

    def test_with_candidates(self):
        fig, ax = plt.subplots()
        plot_utilities(
            qs=self.qs,
            X=self.X_train,
            y=self.y_train,
            **self.qs_dict,
            candidates=self.X_cand,
            ax=ax
        )
        ax.scatter(self.X[:, 0], self.X[:, 1], c="k", marker=".")
        ax.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=self.y_train,
            cmap=self.cmap,
            alpha=0.9,
            marker=".",
        )
        plot_decision_boundary(self.clf, self.bound, ax=ax, cmap=self.cmap)

        fig.savefig(self.path_prefix + "dec_bound_w_cand.pdf")
        comparison = compare_images(
            self.path_prefix + "dec_bound_w_cand_expected.pdf",
            self.path_prefix + "dec_bound_w_cand.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)

    def test_multi_class(self):
        random_state = np.random.RandomState(0)
        X, y = make_classification(
            n_features=2,
            n_redundant=0,
            random_state=0,
            n_classes=3,
            n_clusters_per_class=1,
        )
        train_indices = random_state.randint(0, len(X), size=20)
        cand_indices = np.setdiff1d(np.arange(len(X)), train_indices)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_cand = X[cand_indices]
        clf = ParzenWindowClassifier()
        clf.fit(X_train, y_train)
        qs = UncertaintySampling(random_state=0)
        bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]

        fig, ax = plt.subplots()
        plot_utilities(
            qs=qs, X=X_train, y=y_train, clf=clf, feature_bound=bound, ax=ax
        )
        ax.scatter(X_cand[:, 0], X_cand[:, 1], c="k", marker=".")
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=self.cmap,
            alpha=0.9,
            marker=".",
        )
        plot_decision_boundary(clf, bound, ax=ax, res=101, cmap=self.cmap)
        fig.savefig(self.path_prefix + "dec_bound_multiclass.pdf")
        comparison = compare_images(
            self.path_prefix + "dec_bound_multiclass.pdf",
            self.path_prefix + "dec_bound_multiclass_expected.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)

    def test_svc(self):
        svc = LinearSVC()
        svc.fit(self.X_train, self.y_train)

        fig, ax = plt.subplots()
        plot_utilities(
            qs=self.qs,
            **self.qs_dict,
            X=self.X_train,
            y=self.y_train,
            candidates=self.X_cand,
            ax=ax
        )
        ax.scatter(self.X[:, 0], self.X[:, 1], c="k", marker=".")
        ax.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=self.y_train,
            cmap=self.cmap,
            alpha=0.9,
            marker=".",
        )
        plot_decision_boundary(svc, self.bound, ax=ax, cmap=self.cmap)

        fig.savefig(self.path_prefix + "dec_bound_svc.pdf")
        comparison = compare_images(
            self.path_prefix + "dec_bound_svc_expected.pdf",
            self.path_prefix + "dec_bound_svc.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)

    def test_multi_with_axes(self):
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        query_params_dict = {"clf": self.clf}
        plot_annotator_utilities(
            qs=qs,
            X=self.X,
            y=self.y_active_multi,
            feature_bound=self.bound,
            axes=axes,
            query_params_dict=query_params_dict,
        )

        fig.savefig(self.path_prefix + "multi_with_axes.pdf")
        comparison = compare_images(
            self.path_prefix + "multi_with_axes_expected.pdf",
            self.path_prefix + "multi_with_axes.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)

    def test_multi_without_axes(self):
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        query_params_dict = {"clf": self.clf}
        plot_annotator_utilities(
            qs=qs,
            X=self.X,
            y=self.y_active_multi,
            feature_bound=self.bound,
            query_params_dict=query_params_dict,
        )

        plt.savefig(self.path_prefix + "multi_without_axes.pdf")
        comparison = compare_images(
            self.path_prefix + "multi_without_axes_expected.pdf",
            self.path_prefix + "multi_without_axes.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)

    def test_multi_without_axes_cand(self):
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        query_params_dict = {"clf": self.clf}
        plot_annotator_utilities(
            qs=qs,
            X=self.X,
            candidates=[1, 2, 3],
            y=self.y_active_multi,
            feature_bound=self.bound,
            query_params_dict=query_params_dict,
        )

        plt.savefig(self.path_prefix + "multi_without_axes_cand.pdf")
        comparison = compare_images(
            self.path_prefix + "multi_without_axes_cand_expected.pdf",
            self.path_prefix + "multi_without_axes_cand.pdf",
            tol=0,
        )
        self.assertIsNone(comparison)


class TestClassifier(ClassifierMixin):
    pass
