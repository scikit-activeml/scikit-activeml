import os
import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.utils._testing import assert_allclose

from skactiveml import visualization
from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.exceptions import MappingError
from skactiveml.pool import (
    LabelCardinalityInconsistency,
    MaxLossReductionMaxConfidence,
    TypiClust,
    UncertaintySampling,
    RandomSampling,
    SubSamplingWrapper,
    ValueOfInformationEER,
)
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.regressor import SklearnRegressor
from skactiveml.utils import TargetSpec
from skactiveml.visualization import (
    plot_decision_boundary,
    plot_utilities,
    plot_contour_for_samples,
    plot_annotator_utilities,
    plot_stream_training_data,
    plot_stream_decision_boundary,
)
from skactiveml.visualization._feature_space import (
    _general_plot_utilities,
    _resolve_utility_target_type,
)

# PDF rasterization differs slightly across supported Matplotlib versions.
IMAGE_COMPARE_TOL = 6.0


class TestFeatureSpace(unittest.TestCase):
    def setUp(self):
        self.path_prefix = (
            os.path.dirname(visualization.__file__) + "/tests/images/"
        )
        self.X, self.y = make_classification(
            n_features=2, n_redundant=0, random_state=0
        )
        self.X_stream, self.y_stream = make_blobs(
            n_features=2,
            centers=[[0], [-3], [1], [2], [-0.5]],
            cluster_std=0.7,
            random_state=0,
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
        self.X_train_stream = self.X_stream[train_indices]
        self.y_train_stream = self.y_stream[train_indices]
        self.y_train_multi = np.tile(self.y_train, [5, 1]).T
        self.X_cand = self.X[cand_indices]
        self.clf = ParzenWindowClassifier(random_state=0)
        self.clf.fit(self.X_train, self.y_train)
        self.clf_stream = ParzenWindowClassifier(random_state=0)
        self.clf_stream.fit(self.X_train_stream, self.y_train_stream)
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
        self.stream_bound = [
            [0, len(self.X_stream)],
            [min(self.X_stream), max(self.X_stream)],
        ]
        self.queried_indices = [True] * len(self.y)

        self.cmap = "jet"

        testing.set_font_settings_for_testing()
        testing.set_reproducibility_for_testing()
        testing.setup()

    def tearDown(self):
        plt.close("all")

    def assertImagesClose(self, expected, actual):
        comparison = compare_images(
            self.path_prefix + expected,
            self.path_prefix + actual,
            tol=IMAGE_COMPARE_TOL,
        )
        self.assertIsNone(comparison)

    @staticmethod
    def multilabel_pool():
        X, clusters = make_blobs(
            n_samples=30,
            centers=4,
            n_features=2,
            random_state=0,
        )
        y_true = np.column_stack(
            [
                np.isin(clusters, [0, 1]),
                np.isin(clusters, [1, 2]),
                np.isin(clusters, [2, 3]),
            ]
        ).astype(float)
        y = np.full_like(y_true, np.nan)
        y[:15] = y_true[:15]
        feature_bound = [X.min(axis=0), X.max(axis=0)]
        return X, y, y_true, feature_bound

    @staticmethod
    def multilabel_classifier(proba_format="array"):
        return SklearnClassifier(
            RandomForestClassifier(n_estimators=5, random_state=0),
            classes=[[0, 1], [0, 1], [0, 1]],
            target_type="multi-label",
            proba_format=proba_format,
            random_state=0,
        )

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
        clf.target_spec_ = TargetSpec(
            task="classification",
            target_type="multi-label",
            annotation_type="single-annotator",
            classes=((0, 1), (0, 1)),
        )
        with self.assertRaisesRegex(AttributeError, "multi-label"):
            plot_decision_boundary(
                clf=clf,
                feature_bound=self.bound,
                ax=plt.subplots(1, 2)[1],
                confidence=None,
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

    def test_decision_boundary_multilabel_axes_and_proba_formats(self):
        for proba_format in ["array", "list"]:
            with self.subTest(proba_format=proba_format):
                clf = MultilabelTestClassifier(proba_format=proba_format)
                _, axes = plt.subplots(1, 2)

                returned_axes = plot_decision_boundary(
                    clf,
                    [[0, 0], [1, 1]],
                    ax=axes,
                    res=5,
                    confidence=None,
                )

                self.assertIs(returned_axes, axes)
                vertical = axes[0].collections[0].get_paths()[0].vertices
                horizontal = axes[1].collections[0].get_paths()[0].vertices
                np.testing.assert_allclose(vertical[:, 0], 0.5)
                np.testing.assert_allclose(horizontal[:, 1], 0.5)

    def test_decision_boundary_sklearn_multilabel_proba_formats(self):
        X, _, y_true, feature_bound = self.multilabel_pool()
        for proba_format in ["array", "list"]:
            with self.subTest(proba_format=proba_format):
                clf = self.multilabel_classifier(proba_format).fit(X, y_true)
                _, axes = plt.subplots(1, y_true.shape[1])

                returned_axes = plot_decision_boundary(
                    clf,
                    feature_bound,
                    ax=axes,
                    res=5,
                    confidence=None,
                )

                self.assertIs(returned_axes, axes)
                self.assertTrue(all(ax.collections for ax in axes))

    def test_decision_boundary_rejects_ambiguous_probability_list(self):
        with self.assertRaisesRegex(ValueError, "target_spec_"):
            plot_decision_boundary(
                AmbiguousListClassifier(),
                [[0, 0], [1, 1]],
                confidence=None,
            )

    def test_decision_boundary_multilabel_overlay_and_confidence(self):
        clf = MultilabelTestClassifier(proba_format="array")
        _, ax = plt.subplots()

        returned_ax = plot_decision_boundary(
            clf,
            [[0, 0], [1, 1]],
            ax=ax,
            res=5,
            confidence=0.75,
        )

        self.assertIs(returned_ax, ax)
        self.assertEqual(len(ax.collections), 4)
        np.testing.assert_allclose(ax.collections[1].levels, [0.25, 0.75])
        np.testing.assert_allclose(ax.collections[3].levels, [0.25, 0.75])
        self.assertFalse(
            np.array_equal(
                ax.collections[0].get_edgecolor(),
                ax.collections[2].get_edgecolor(),
            )
        )

    def test_decision_boundary_multilabel_axes_confidence_colors(self):
        clf = MultilabelTestClassifier(proba_format="array")
        _, axes = plt.subplots(1, 2)

        plot_decision_boundary(
            clf,
            [[0, 0], [1, 1]],
            ax=axes,
            res=5,
            boundary_dict={"colors": "black"},
            confidence=0.75,
        )

        confidence_colors = np.array(
            [
                plt.colormaps["coolwarm"](0.0),
                plt.colormaps["coolwarm"](1.0),
            ]
        )
        confidence_colors[:, 3] = 0.9
        for ax in axes:
            assert_allclose(
                ax.collections[0].get_edgecolor(), [[0.0, 0.0, 0.0, 1.0]]
            )
            assert_allclose(
                ax.collections[1].get_edgecolor(), confidence_colors
            )

    def test_decision_boundary_multilabel_axes_count(self):
        clf = MultilabelTestClassifier(proba_format="array")
        _, axes = plt.subplots(1, 3)

        with self.assertRaisesRegex(ValueError, "each label output"):
            plot_decision_boundary(
                clf,
                [[0, 0], [1, 1]],
                ax=axes,
                confidence=None,
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
            feature_bound=self.bound,
        )

    def test__general_plot_utilities_param_X(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=np.ones([len(self.X), 3]),
            y=self.y,
            **self.qs_dict,
            feature_bound=self.bound,
        )

    def test__general_plot_utilities_param_y(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=np.zeros(len(self.y) + 1),
            **self.qs_dict,
            feature_bound=self.bound,
        )

    def test__general_plot_utilities_param_candidates(self):
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y,
            **self.qs_dict,
            candidates=[100],
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
            feature_bound=self.bound,
        )

    def test__general_plot_utilities_param_plot_annotators(self):
        self.assertRaises(
            TypeError,
            _general_plot_utilities,
            qs=self.qs,
            X=self.X,
            y=self.y,
            **self.qs_dict,
            plot_annotators=[4],
        )
        _, axes = plt.subplots(1, 2)
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=qs,
            X=self.X,
            y=self.y_active_multi,
            **self.qs_dict,
            plot_annotators=[4],
            axes=axes,
        )

    def test__general_plot_utilities_param_ignore_undefined_query_params(self):
        _general_plot_utilities(
            qs=ValueOfInformationEER(),
            X=self.X,
            y=self.y_active,
            **self.qs_dict,
            ignore_undefined_query_params=True,
            feature_bound=self.bound,
        )
        _general_plot_utilities(
            qs=self.qs,
            X=self.X,
            y=self.y,
            candidates=None,
            **self.qs_dict,
            ignore_undefined_query_params=True,
            feature_bound=self.bound,
        )
        _general_plot_utilities(
            qs=self.qs,
            X=self.X,
            y=self.y,
            candidates=[1],
            **self.qs_dict,
            ignore_undefined_query_params=True,
            feature_bound=self.bound,
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
            res=-3,
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
            ax=2,
        )
        _, axes = plt.subplots(1, 2)
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        self.assertRaises(
            ValueError,
            _general_plot_utilities,
            qs=qs,
            X=self.X,
            y=self.y_active_multi,
            **self.qs_dict,
            feature_bound=self.bound,
            ax=axes,
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
            axes=2,
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
            contour_dict="string",
        )
        _general_plot_utilities(
            qs=self.qs,
            **self.qs_dict,
            X=self.X,
            y=self.y,
            feature_bound=self.bound,
            contour_dict={"linestyles": "."},
        )

    def test_plot_utilities_multilabel_prediction_strategy(self):
        X, y, _, feature_bound = self.multilabel_pool()
        clf = self.multilabel_classifier()
        qs = LabelCardinalityInconsistency(
            target_type="multi-label", random_state=0
        )
        _, ax = plt.subplots()

        returned_ax = plot_utilities(
            qs,
            X,
            y,
            clf=clf,
            feature_bound=feature_bound,
            ax=ax,
            res=7,
        )

        self.assertIs(returned_ax, ax)
        self.assertGreater(len(ax.collections), 0)

    def test_plot_utilities_multilabel_probability_strategy(self):
        X, y, _, feature_bound = self.multilabel_pool()
        clf = self.multilabel_classifier()
        qs = MaxLossReductionMaxConfidence(
            target_type="multi-label", random_state=0
        )
        discriminator = ParzenWindowClassifier(random_state=0)
        _, ax = plt.subplots()

        returned_ax = plot_utilities(
            qs,
            X,
            y,
            clf=clf,
            discriminator=discriminator,
            feature_bound=feature_bound,
            ax=ax,
            res=7,
        )

        self.assertIs(returned_ax, ax)
        self.assertGreater(len(ax.collections), 0)

    def test_plot_utilities_multilabel_mapping_fallback(self):
        X, y, _, feature_bound = self.multilabel_pool()
        qs = TypiClust(
            target_type="multi-label",
            cluster_algo_dict={"n_init": 1},
            random_state=0,
        )
        _, ax = plt.subplots()

        returned_ax = plot_utilities(
            qs,
            X,
            y,
            feature_bound=feature_bound,
            ax=ax,
            res=7,
        )

        self.assertIs(returned_ax, ax)
        self.assertGreater(len(ax.collections), 0)

        wrapper = SubSamplingWrapper(qs, max_candidates=10, random_state=0)
        _, wrapper_ax = plt.subplots()

        returned_wrapper_ax = plot_utilities(
            wrapper,
            X,
            y,
            feature_bound=feature_bound,
            ax=wrapper_ax,
            res=7,
        )

        self.assertIs(returned_wrapper_ax, wrapper_ax)
        self.assertGreater(len(wrapper_ax.collections), 0)

    def test_plot_utilities_regression_column_mapping_fallback(self):
        X, _ = make_blobs(
            n_samples=30,
            centers=4,
            n_features=2,
            random_state=0,
        )
        y = np.full((len(X), 1), np.nan)
        y[:15, 0] = X[:15, 0]
        reg = SklearnRegressor(
            DecisionTreeRegressor(random_state=0),
            target_type="auto",
            random_state=0,
        ).fit(X, y)
        qs = MappingOnlyRegressionStrategy(target_type="auto", random_state=0)
        feature_bound = [X.min(axis=0), X.max(axis=0)]
        _, ax = plt.subplots()

        returned_ax = plot_utilities(
            qs,
            X,
            y,
            reg=reg,
            feature_bound=feature_bound,
            ax=ax,
            res=7,
        )

        self.assertIs(returned_ax, ax)
        self.assertGreater(len(ax.collections), 0)

    def test_resolve_utility_target_type_of_multi_annotator_strategy(self):
        qs = SingleAnnotatorWrapper(RandomSampling(), random_state=0)

        self.assertEqual(
            _resolve_utility_target_type(qs, self.y_active_multi, {}),
            "single-output",
        )

    def test_plot_utilities_rejects_annotator_options_for_multilabel(self):
        y = np.column_stack([self.y_active, self.y_active])
        qs = RandomSampling(target_type="multi-label", random_state=0)
        _, axes = plt.subplots(1, 2)

        with self.assertRaisesRegex(TypeError, "`axes`"):
            plot_utilities(qs, self.X, y, axes=axes)
        with self.assertRaisesRegex(TypeError, "plot_annotator"):
            plot_utilities(qs, self.X, y, plot_annotators=[0])

    # Tests for plot_stream_decision_boundary function
    def test_plot_stream_decision_boundary_param_ax(self):
        self.assertRaises(
            TypeError,
            plot_stream_decision_boundary,
            t_x=0,
            plot_step=1,
            clf=self.clf,
            X=self.X_stream,
            pred_list=[],
            ax=2,
        )
        _, axes = plt.subplots(1, 2)
        self.assertRaises(
            TypeError,
            plot_stream_decision_boundary,
            t_x=0,
            plot_step=1,
            clf=self.clf,
            X=self.X_stream,
            pred_list=[],
            ax=axes,
        )

    def test_plot_stream_decision_boundary_param_t_x(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_decision_boundary,
            ax=ax,
            t_x=-1,
            plot_step=1,
            clf=self.clf,
            X=self.X_stream,
            pred_list=[],
        )

    def test_plot_stream_decision_boundary_param_plot_step(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_decision_boundary,
            ax=ax,
            t_x=0,
            plot_step=0,
            clf=self.clf,
            X=self.X_stream,
            pred_list=[],
        )

    def test_plot_stream_decision_boundary_param_clf(self):
        _, ax = plt.subplots()
        self.assertRaises(
            TypeError,
            plot_stream_decision_boundary,
            clf=self.qs,
            ax=ax,
            t_x=0,
            plot_step=1,
            X=self.X_stream,
            pred_list=[],
        )
        clf = TestClassifier()
        self.assertRaises(
            AttributeError,
            plot_stream_decision_boundary,
            clf=clf,
            ax=ax,
            t_x=0,
            plot_step=1,
            X=self.X_stream,
            pred_list=[],
        )

    def test_plot_stream_decision_boundary_param_X(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_decision_boundary,
            ax=ax,
            t_x=0,
            plot_step=1,
            pred_list=[],
            clf=self.clf,
            X=np.ones([len(self.X), 3]),
        )

    def test_plot_stream_decision_boundary_param_pred_list(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_decision_boundary,
            ax=ax,
            t_x=0,
            plot_step=1,
            pred_list=True,
            clf=self.clf,
            X=self.X_stream,
        )

    def test_plot_stream_decision_boundary_param_color(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_decision_boundary,
            ax=ax,
            t_x=0,
            plot_step=1,
            pred_list=[],
            clf=self.clf,
            X=self.X_stream,
            color=0,
        )

    def test_plot_stream_decision_boundary_param_res(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_decision_boundary,
            ax=ax,
            t_x=0,
            plot_step=1,
            pred_list=[],
            clf=self.clf,
            X=self.X_stream,
            res=3,
        )

    # Tests for plot_stream_training_data function
    def test_plot_stream_training_data_param_X(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            queried_indices=self.queried_indices,
            X=np.ones([len(self.X), 3]),
            y=self.y_stream,
        )

    def test_plot_stream_training_data_param_y(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            queried_indices=self.queried_indices,
            X=self.X_stream,
            y=np.zeros(len(self.y_stream) + 1),
        )

    def test_plot_stream_training_data_param_queried_indices(self):
        _, ax = plt.subplots()
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=True,
            y=self.y_stream,
        )

    def test_plot_stream_training_data_param_classes(self):
        _, ax = plt.subplots()
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            ax=ax,
            classes=1,
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=self.queried_indices,
            y=self.y_stream,
        )

    def test_plot_stream_training_data_unlabeled_color(self):
        _, ax = plt.subplots()
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=self.queried_indices,
            y=self.y_stream,
            unlabeled_color=1,
        )

    def test_plot_stream_training_data_cmap(self):
        _, ax = plt.subplots()
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=self.queried_indices,
            y=self.y_stream,
            cmap=True,
        )

    def test_plot_stream_training_data_alpha(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=self.queried_indices,
            y=self.y_stream,
            alpha=-1,
        )

    def test_plot_stream_training_data_linewidth(self):
        _, ax = plt.subplots()
        self.assertRaises(
            ValueError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=self.queried_indices,
            y=self.y_stream,
            linewidth="string",
        )

    def test_plot_stream_training_data_plot_cand_highlight(self):
        _, ax = plt.subplots()
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            ax=ax,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            queried_indices=self.queried_indices,
            y=self.y_stream,
            plot_cand_highlight="True",
        )

    def test_plot_stream_training_data_param_ax(self):
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            y=self.y_stream,
            ax=2,
        )
        _, axes = plt.subplots(1, 2)
        self.assertRaises(
            TypeError,
            plot_stream_training_data,
            classes=np.unique(self.y),
            feature_bound=self.stream_bound,
            X=self.X_stream,
            y=self.y_stream,
            ax=axes,
        )

    def test_plot_contour_for_samples_param_X(self):
        for X in [None, 1, np.arange(10)]:
            self.assertRaises(
                ValueError,
                plot_contour_for_samples,
                X=X,
                values=self.utilities,
            )
        values = self.utilities.copy()
        values[:5] = -np.inf
        values[5:] = np.inf
        plot_contour_for_samples(values=values, X=self.X)

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
        self.assertImagesClose(
            "dec_bound_wo_cand_expected.pdf", "dec_bound_wo_cand.pdf"
        )

    def test_with_candidates(self):
        fig, ax = plt.subplots()
        plot_utilities(
            qs=self.qs,
            X=self.X_train,
            y=self.y_train,
            **self.qs_dict,
            candidates=self.X_cand,
            ax=ax,
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
        self.assertImagesClose(
            "dec_bound_w_cand_expected.pdf", "dec_bound_w_cand.pdf"
        )

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
        self.assertImagesClose(
            "dec_bound_multiclass_expected.pdf", "dec_bound_multiclass.pdf"
        )

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
            ax=ax,
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
        self.assertImagesClose(
            "dec_bound_svc_expected.pdf", "dec_bound_svc.pdf"
        )

    def test_multi_with_axes(self):
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        plot_annotator_utilities(
            qs=qs,
            X=self.X,
            y=self.y_active_multi,
            feature_bound=self.bound,
            axes=axes,
            clf=self.clf,
        )

        fig.savefig(self.path_prefix + "multi_with_axes.pdf")
        self.assertImagesClose(
            "multi_with_axes_expected.pdf", "multi_with_axes.pdf"
        )

    def test_multi_without_axes(self):
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        plot_annotator_utilities(
            qs=qs,
            X=self.X,
            y=self.y_active_multi,
            feature_bound=self.bound,
            clf=self.clf,
        )

        plt.savefig(self.path_prefix + "multi_without_axes.pdf")
        self.assertImagesClose(
            "multi_without_axes_expected.pdf", "multi_without_axes.pdf"
        )

    def test_multi_without_axes_cand(self):
        qs = SingleAnnotatorWrapper(clone(self.qs), random_state=0)
        plot_annotator_utilities(
            qs=qs,
            X=self.X,
            candidates=[1, 2, 3],
            y=self.y_active_multi,
            feature_bound=self.bound,
            clf=self.clf,
        )

        plt.savefig(self.path_prefix + "multi_without_axes_cand.pdf")
        self.assertImagesClose(
            "multi_without_axes_cand_expected.pdf",
            "multi_without_axes_cand.pdf",
        )

    def test_stream(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, len(self.X_stream))
        ax.set_ylim(bottom=min(self.X_stream), top=max(self.X_stream))
        t_x = len(self.y_stream)
        res = 25
        pred_list = [np.full(res, fill_value=0)]

        # add predictions as z needs at least a (2, 2) array
        np.random.seed(0)
        predictions = np.random.choice(a=[0, 1], size=res, p=[0.5, 0.5])
        pred_list.append(predictions)

        ax, pred_list = plot_stream_decision_boundary(
            ax,
            t_x=len(self.y_stream),
            plot_step=t_x // 2,
            clf=self.clf_stream,
            X=self.X_stream,
            pred_list=pred_list,
            res=res,
        )
        p = 0.2
        np.random.seed(0)
        queried_indices = np.random.choice(
            a=[True, False], size=len(self.y_stream), p=[p, 1 - p]
        )

        _ = plot_stream_training_data(
            ax,
            self.X_stream,
            self.y_stream,
            queried_indices=queried_indices,
            classes=np.unique(self.y_train),
            feature_bound=self.stream_bound,
        )

        fig.savefig(self.path_prefix + "dec_bound_w_cand_stream.pdf")
        self.assertImagesClose(
            "dec_bound_w_cand_stream_expected.pdf",
            "dec_bound_w_cand_stream.pdf",
        )


class TestClassifier(ClassifierMixin):
    pass


class MultilabelTestClassifier(ClassifierMixin):
    def __init__(self, proba_format):
        self.proba_format = proba_format
        self.target_spec_ = TargetSpec(
            task="classification",
            target_type="multi-label",
            annotation_type="single-annotator",
            classes=((0, 1), (0, 1)),
        )

    def predict_proba(self, X):
        positive_probas = np.column_stack([X[:, 0], X[:, 1]])
        if self.proba_format == "array":
            return positive_probas
        return [
            np.column_stack([1 - positive_probas[:, j], positive_probas[:, j]])
            for j in range(positive_probas.shape[1])
        ]


class AmbiguousListClassifier(ClassifierMixin):
    def predict_proba(self, X):
        positive_probas = np.column_stack([X[:, 0], X[:, 1]])
        return [
            np.column_stack([1 - positive_probas[:, j], positive_probas[:, j]])
            for j in range(positive_probas.shape[1])
        ]


class MappingOnlyRegressionStrategy(SingleAnnotatorPoolQueryStrategy):
    @property
    def _target_capabilities(self):
        return frozenset({("regression", "single-output", "single-annotator")})

    def query(
        self,
        X,
        y,
        reg,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        if reg.target_spec_.target_type != "single-output":
            raise ValueError("`reg` must have single-output target semantics.")
        candidates = np.asarray(candidates)
        if candidates.ndim == 2:
            raise MappingError("This strategy requires candidate indices.")
        utilities = np.full((batch_size, len(X)), np.nan)
        utilities[:, candidates] = 1.0
        query_indices = candidates[:batch_size]
        if return_utilities:
            return query_indices, utilities
        return query_indices
