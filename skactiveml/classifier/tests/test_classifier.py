import inspect
import unittest
from copy import deepcopy
from importlib import import_module
from os import path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import NotFittedError

from skactiveml import classifier
from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import multiannotator
from skactiveml.utils import call_func


class TestClassifier(unittest.TestCase):
    def setUp(self):
        # Build dictionary of attributes.
        self.classifiers = {}
        self.is_multi = {}
        for clf_name in classifier.__all__:
            clf = getattr(classifier, clf_name)
            if inspect.isclass(clf) and issubclass(clf, SkactivemlClassifier):
                self.classifiers[clf_name] = clf
                self.is_multi[clf_name] = False
        for clf_name in multiannotator.__all__:
            clf = getattr(multiannotator, clf_name)
            if inspect.isclass(clf) and issubclass(clf, SkactivemlClassifier):
                self.classifiers[clf_name] = clf
                self.is_multi[clf_name] = True

        # Create data set for testing.
        self.X, self.y_true = make_blobs(random_state=0)
        self.classes = np.unique(self.y_true)
        self.missing_label = None
        self.X = StandardScaler().fit_transform(self.X)
        self.y_single = self.y_true.copy()
        self.y_single = self.y_single.astype("object")
        self.y_single[:20] = self.missing_label
        self.y_single_missing_label = np.full_like(
            self.y_single, self.missing_label, dtype=object
        )
        self.y_multi = np.repeat(self.y_true.reshape(-1, 1), 3, axis=1)
        self.y_multi = self.y_multi.astype("object")
        self.y_multi[:100, 0] = self.missing_label
        self.y_multi[200:, 1] = self.missing_label
        self.y_multi_missing_label = np.full_like(
            self.y_multi, self.missing_label, dtype=object
        )

        # Set up estimators for Sklearn wrapper and multi-annot classifier.
        self.estimator = GaussianNB()
        cmm = classifier.MixtureModelClassifier(
            missing_label=self.missing_label, random_state=0
        )
        gnb = classifier.SklearnClassifier(
            GaussianNB(), missing_label=self.missing_label
        )
        mlp = classifier.SklearnClassifier(
            MLPClassifier(), missing_label=self.missing_label
        )
        self.estimators = [
            ("MixtureModelClassifier", cmm),
            ("GaussianNB", gnb),
            ("MLP", mlp),
        ]

    def test_classifiers(self):
        # Test predictions of classifiers.
        for clf in self.classifiers:
            self.y = self.y_single
            self.y_missing_label = self.y_single_missing_label
            self.y_shape = self.y_multi
            if self.is_multi[clf]:
                self.y = self.y_multi
                self.y_missing_label = self.y_multi_missing_label
                self.y_shape = self.y_single
            self._test_single_classifier(clf)

    def _test_single_classifier(self, clf):
        # Create fully initialized classifier.
        clf_mdl = call_func(
            self.classifiers[clf],
            estimator=self.estimator,
            estimators=self.estimators,
            classes=self.classes,
            cost_matrix=1 - np.eye(len(self.classes)),
            missing_label=self.missing_label,
            voting="soft",
            random_state=0,
        )

        # Create classifier without classes parameter.
        clf_mdl_cls = deepcopy(clf_mdl)
        clf_mdl_cls.classes = None

        # Test classifier without fitting.
        with self.subTest(msg="Not Fitted Test", clf_name=clf):
            self.assertRaises(NotFittedError, clf_mdl_cls.predict, X=self.X)
            self.assertRaises(NotFittedError, clf_mdl.predict, X=self.X)

        # Test classifier on empty data set.
        with self.subTest(msg="Empty Data Test", clf_name=clf):
            self.assertRaises(ValueError, clf_mdl_cls.predict, X=self.X)
            clf_mdl.fit(X=[], y=[])
            P = clf_mdl.predict_proba(X=self.X)
            np.testing.assert_array_equal(P, np.ones((len(self.X), 3)) / 3)
            if hasattr(clf_mdl, "predict_annotator_perf"):
                P = clf_mdl.predict_annotator_perf(X=self.X)
                np.testing.assert_array_equal(P, np.ones((len(self.X), 1)) / 3)

        with self.subTest(msg="Labels Shape Test", clf_name=clf):
            self.assertRaises(
                ValueError, clf_mdl.fit, X=self.X, y=self.y_shape
            )

        # Test classifier on data with only missing labels.
        with self.subTest(msg="Missing Label Test", clf_name=clf):
            self.assertRaises(
                ValueError, clf_mdl_cls.fit, X=self.X, y=self.y_missing_label
            )
            clf_mdl.fit(X=self.X, y=self.y_missing_label)
            score = clf_mdl.score(self.X, self.y_true)
            self.assertTrue(score > 0)
            P_exp = np.ones((len(self.X), len(self.classes))) / len(
                self.classes
            )
            P = clf_mdl.predict_proba(self.X)
            np.testing.assert_array_equal(P_exp, P)
            if hasattr(clf_mdl, "predict_freq"):
                F_exp = np.zeros((len(self.X), len(self.classes)))
                F = clf_mdl.predict_freq(self.X)
                np.testing.assert_array_equal(F_exp, F)

        # Test classifier on full data set.
        with self.subTest(msg="Full Data Test", clf_name=clf):
            score = clf_mdl.fit(self.X, self.y).score(self.X, self.y_true)
            self.assertTrue(score > 0.8)
            if hasattr(clf_mdl, "predict_proba"):
                P = clf_mdl.predict_proba(self.X)
                self.assertTrue(np.sum(P != 1 / len(self.classes)) > 0)
            if hasattr(clf_mdl, "predict_freq"):
                F = clf_mdl.predict_freq(self.X)
                self.assertTrue(np.sum(F) > 0)

    def test_param(self):
        not_test = [
            "self",
            "kwargs",
            "classes",
            "X",
            "y",
            "sample_weight",
            "missing_label",
            "random_state",
            "cost_matrix",
            "class_prior",
        ]
        for clf in self.classifiers:
            self.y = self.y_single
            self.y_missing_label = self.y_single_missing_label
            self.y_shape = self.y_multi
            mod = "skactiveml.classifier.tests.test"
            if self.is_multi[clf]:
                self.y = self.y_multi
                mod = "skactiveml.classifier.multiannotator.tests.test"
                self.y_missing_label = self.y_multi_missing_label
                self.y_shape = self.y_single
            self._test_single_classifier(clf)
            with self.subTest(msg=f"Parameter/Method Test", clf=clf):
                # Get initial parameters.
                clf_class = self.classifiers[clf]
                init_params = inspect.signature(clf_class).parameters.keys()
                init_params = list(init_params)

                # Check initial parameters.
                values = [Dummy() for i in range(len(init_params))]
                clf_mdl = clf_class(*values)
                for param, value in zip(init_params, values):
                    self.assertTrue(
                        hasattr(clf_mdl, param),
                        msg='"{}" not tested for __init__().'.format(param),
                    )
                    self.assertEqual(getattr(clf_mdl, param), value)

                # Get test class to check.
                class_filename = path.basename(inspect.getfile(clf_class))[:-3]
                mod += class_filename
                mod = import_module(mod)
                test_class_name = "Test" + clf_class.__name__
                self.assertTrue(
                    hasattr(mod, test_class_name),
                    msg="{} has no test called {}.".format(
                        clf, test_class_name
                    ),
                )
                test_obj = getattr(mod, test_class_name)

                # Check init parameters.
                for param in np.setdiff1d(init_params, not_test):
                    test_func_name = "test_init_param_" + param
                    msg = (
                        f"'{test_func_name}()' missing for parameter "
                        f"'{param}' of __init__() from {clf}."
                    )
                    self.assertTrue(hasattr(test_obj, test_func_name), msg)

                methods = [
                    "fit",
                    "predict",
                    "predict_proba",
                    "predict_freq",
                    "predict_annotator_perf",
                ]
                for m in methods:
                    if not hasattr(clf_mdl, m):
                        continue
                    test_func_name = f"test_{m}"
                    msg = f"'{test_func_name}()' in test of {clf}."
                    self.assertTrue(hasattr(test_obj, test_func_name), msg)

                # Check standard parameters of __init__ method.
                self._test_init_param_random_state(clf_class)
                self._test_init_param_classes(clf_class)
                self._test_init_param_missing_label(clf_class)
                self._test_init_param_cost_matrix(clf_class)
                if hasattr(clf_mdl, "predict_freq"):
                    self._test_init_param_class_prior(clf_class)

                # Check standard parameters of fit method.
                self._test_fit_param_X(clf_class)
                self._test_fit_param_y(clf_class)
                fit_params = inspect.signature(clf_class.fit).parameters.keys()
                fit_params = list(fit_params)
                if "sample_weight" in fit_params:
                    self._test_fit_param_sample_weight(clf_class)
                self._test_predict_param_X(clf_class)
                if hasattr(clf_mdl, "predict_proba"):
                    self._test_predict_proba_param_X(clf_class)
                if hasattr(clf_mdl, "predict_freq"):
                    self._test_predict_freq_param_X(clf_class)
                if hasattr(clf_mdl, "predict_annotator_perf"):
                    self._test_predict_annot_perf_param_X(clf_class)

    def _test_init_param_class_prior(self, clf_class):
        clf_mdl = call_func(
            clf_class, estimator=self.estimator, estimators=self.estimators
        )
        self.assertEqual(clf_mdl.class_prior, 0)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            class_prior=2,
        )
        self.assertEqual(clf_mdl.class_prior, 2)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
            class_prior=-1.0,
        )
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
            class_prior=["test"],
        )
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
            class_prior="test",
        )
        self.assertRaises(TypeError, clf_mdl.fit, X=self.X, y=self.y)

    def _test_init_param_classes(self, clf_class):
        clf_mdl = call_func(
            clf_class, estimator=self.estimator, estimators=self.estimators
        )
        self.assertEqual(clf_mdl.classes, None)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            classes=[0, 1],
        )
        np.testing.assert_array_equal(clf_mdl.classes, [0, 1])
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            classes="Test",
            missing_label=self.missing_label,
        )
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            classes=[0, 1],
            missing_label=self.missing_label,
        )
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            classes=[0, 1, self.missing_label],
            missing_label=self.missing_label,
        )
        self.assertRaises(TypeError, clf_mdl.fit, X=self.X, y=self.y)

    def _test_init_param_cost_matrix(self, clf_class):
        clf_mdl = call_func(
            clf_class, estimator=self.estimator, estimators=self.estimators
        )
        self.assertEqual(clf_mdl.cost_matrix, None)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            cost_matrix=[1, 2],
            missing_label=self.missing_label,
            classes=self.classes,
        )
        np.testing.assert_array_equal(clf_mdl.cost_matrix, [1, 2])
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            cost_matrix=1 - np.eye(2),
            classes=self.classes,
            missing_label=self.missing_label,
        )
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            cost_matrix=[["2", "5", "3"], ["a", "5", "3"], ["a", "5", "3"]],
            classes=self.classes,
            missing_label=self.missing_label,
        )
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)

    def _test_init_param_missing_label(self, clf_class):
        clf_mdl = call_func(
            clf_class, estimator=self.estimator, estimators=self.estimators
        )
        self.assertTrue(np.isnan(clf_mdl.missing_label))
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label="Test",
        )
        self.assertTrue(clf_mdl.missing_label, "Test")
        self.assertRaises(TypeError, clf_mdl.fit, X=self.X, y=self.y)

    def _test_init_param_random_state(self, clf_class):
        clf_mdl = call_func(
            clf_class, estimator=self.estimator, estimators=self.estimators
        )
        self.assertTrue(clf_mdl.random_state is None)
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            random_state="Test",
            missing_label=self.missing_label,
        )
        self.assertEqual(clf_mdl.random_state, "Test")
        self.assertRaises(ValueError, clf_mdl.fit, X=self.X, y=self.y)

    def _test_fit_param_X(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        y = [0, 1]
        self.assertRaises(ValueError, clf_mdl.fit, X=[0, 1], y=y)
        self.assertRaises(ValueError, clf_mdl.fit, X=[[0], [1], [2]], y=y)

    def _test_fit_param_y(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        X = [[0], [1]]
        self.assertRaises(ValueError, clf_mdl.fit, X=X, y=[0, 1, 2])
        self.assertRaises(ValueError, clf_mdl.fit, X=X, y=[[0], [1], [2]])

    def _test_fit_param_sample_weight(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        X = [[0], [1]]
        y = [0, 1]
        self.assertRaises(
            ValueError, clf_mdl.fit, X=X, y=y, sample_weight=[0, 1, 1]
        )
        self.assertRaises(
            ValueError, clf_mdl.fit, X=X, y=y, sample_weight=[[1, 1], [1, 1]]
        )

    def _test_predict_param_X(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        clf_mdl.fit(X=self.X, y=self.y)
        self.assertRaises(ValueError, clf_mdl.predict, X=[0, 0])
        self.assertRaises(ValueError, clf_mdl.predict, X=[[0], [0]])
        self.assertRaises(ValueError, clf_mdl.predict, X=[["x", "y"]])

    def _test_predict_proba_param_X(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        clf_mdl.fit(X=self.X, y=self.y)
        self.assertRaises(ValueError, clf_mdl.predict_proba, X=[0, 0])
        self.assertRaises(ValueError, clf_mdl.predict_proba, X=[[0], [0]])
        self.assertRaises(ValueError, clf_mdl.predict_proba, X=[["x", "y"]])

    def _test_predict_freq_param_X(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        clf_mdl.fit(X=self.X, y=self.y)
        self.assertRaises(ValueError, clf_mdl.predict_freq, X=[0, 0])
        self.assertRaises(ValueError, clf_mdl.predict_freq, X=[[0], [0]])
        self.assertRaises(ValueError, clf_mdl.predict_freq, X=[["x", "y"]])

    def _test_predict_annot_perf_param_X(self, clf_class):
        clf_mdl = call_func(
            clf_class,
            estimator=self.estimator,
            estimators=self.estimators,
            missing_label=self.missing_label,
        )
        clf_mdl.fit(X=self.X, y=self.y)
        self.assertRaises(ValueError, clf_mdl.predict_annotator_perf, X=[0, 0])
        self.assertRaises(
            ValueError, clf_mdl.predict_annotator_perf, X=[[0], [0]]
        )
        self.assertRaises(
            ValueError, clf_mdl.predict_annotator_perf, X=[["x", "y"]]
        )


class Dummy:
    def __init__(self):
        pass
