import inspect
import json
import os
import shutil
import unittest
import warnings
from importlib import import_module
from os import path, listdir

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from docs.generate import generate_examples
from skactiveml import pool
from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.classifier import (
    ParzenWindowClassifier,
    MixtureModelClassifier,
    SklearnClassifier,
)
from skactiveml.exceptions import MappingError
from skactiveml.pool import FourDs
from skactiveml.utils import (
    call_func,
    is_unlabeled,
    MISSING_LABEL,
    is_labeled,
    unlabeled_indices,
)
from skactiveml.utils._label import check_equal_missing_label


class TestGeneral(unittest.TestCase):
    def setUp(self):
        self.MISSING_LABEL = MISSING_LABEL
        self.X, self.y_true = make_blobs(
            n_samples=10,
            n_features=2,
            centers=2,
            cluster_std=1,
            random_state=1,
        )
        self.budget = 5
        self.clf = ParzenWindowClassifier(
            classes=np.unique(self.y_true),
            missing_label=MISSING_LABEL,
            random_state=0,
        )
        self.cmm = MixtureModelClassifier(
            classes=np.unique(self.y_true),
            missing_label=MISSING_LABEL,
            random_state=0,
        )
        self.ensemble = SklearnClassifier(
            classes=np.unique(self.y_true),
            missing_label=MISSING_LABEL,
            estimator=RandomForestClassifier(random_state=0),
            random_state=0,
        )

        self.y_missing_label = np.full(self.y_true.shape, self.MISSING_LABEL)
        self.y = self.y_true.copy()
        self.y[:3] = self.y_true[:3]
        self.query_strategies = {}
        for qs_name in pool.__all__:
            qs = getattr(pool, qs_name)
            if inspect.isclass(qs) and issubclass(
                    qs, SingleAnnotatorPoolQueryStrategy
            ):
                self.query_strategies[qs_name] = qs
        print(self.query_strategies.keys())

    def test_al_cycle(self):
        for qs_name in self.query_strategies:
            clf = self.cmm if qs_name == "FourDs" else self.clf
            with self.subTest(msg="Random State", qs_name=qs_name):
                y = np.full(self.y_true.shape, self.MISSING_LABEL)
                qs = call_func(
                    self.query_strategies[qs_name],
                    only_mandatory=False,
                    missing_label=self.MISSING_LABEL,
                    classes=np.unique(self.y_true),
                    ensemble=self.ensemble,
                    random_state=np.random.RandomState(0),
                )

                id1, u1 = call_func(
                    qs.query,
                    X=self.X,
                    y=y,
                    clf=clf,
                    X_eval=self.X,
                    ensemble=self.ensemble,
                    return_utilities=True,
                )
                id2, u2 = call_func(
                    qs.query,
                    X=self.X,
                    y=y,
                    clf=clf,
                    X_eval=self.X,
                    ensemble=self.ensemble,
                    return_utilities=True,
                )
                self.assertEqual(len(u1[0]), len(self.X))
                np.testing.assert_array_equal(id1, id2)
                np.testing.assert_array_equal(u1, u2)

            with self.subTest(msg="Batch", qs_name=qs_name):
                y = np.full(self.y_true.shape, self.MISSING_LABEL)
                y[0:2] = self.y_true[0:2]
                qs = call_func(
                    self.query_strategies[qs_name],
                    only_mandatory=True,
                    missing_label=self.MISSING_LABEL,
                    classes=np.unique(self.y_true),
                    ensemble=self.ensemble,
                    random_state=np.random.RandomState(0),
                )

                ids, u = call_func(
                    qs.query,
                    X=self.X,
                    y=y,
                    clf=clf,
                    X_eval=self.X,
                    batch_size=5,
                    ensemble=self.ensemble,
                    return_utilities=True,
                )

                self.assertEqual(len(ids), 5)
                self.assertEqual(
                    len(u), 5, msg="utility score should " "have shape (5xN)"
                )
                self.assertEqual(
                    len(u[0]),
                    len(self.X),
                    msg="utility score must have shape (5xN)",
                )

                unlabeled = np.where(is_unlabeled(y))[0]
                labeled = np.where(is_labeled(y))[0]
                self.assertEqual(sum(np.isnan(u[0][labeled])), len(labeled))
                self.assertEqual(sum(np.isnan(u[0][unlabeled])), 0)

                self.assertWarns(
                    Warning,
                    call_func,
                    f_callable=qs.query,
                    X=self.X,
                    y=y,
                    clf=clf,
                    X_eval=self.X,
                    ensemble=self.ensemble,
                    batch_size=15,
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    ids = call_func(
                        qs.query,
                        X=self.X,
                        y=y,
                        clf=clf,
                        X_eval=self.X,
                        ensemble=self.ensemble,
                        batch_size=15,
                    )
                    self.assertEqual(len(ids), len(unlabeled))

            with self.subTest(msg="Candidate Variants", qs_name=qs_name):
                y = np.full(self.y_true.shape, self.MISSING_LABEL)
                y[0:5] = self.y_true[0:5]
                qs = call_func(
                    self.query_strategies[qs_name],
                    only_mandatory=False,
                    missing_label=self.MISSING_LABEL,
                    classes=np.unique(self.y_true),
                    ensemble=self.ensemble,
                    random_state=np.random.RandomState(0),
                )

                unld_idx = unlabeled_indices(y, self.MISSING_LABEL)
                ids1, u1 = call_func(
                    qs.query,
                    X=self.X,
                    y=y,
                    clf=clf,
                    X_eval=self.X,
                    ensemble=self.ensemble,
                    return_utilities=True,
                )
                ids2, u2 = call_func(
                    qs.query,
                    X=self.X,
                    y=y,
                    clf=clf,
                    X_eval=self.X,
                    candidates=unld_idx,
                    ensemble=self.ensemble,
                    return_utilities=True,
                )
                np.testing.assert_array_equal(u1, u2)

                if not isinstance(qs, FourDs):
                    unld_idx = unlabeled_indices(y, self.MISSING_LABEL)[:2]
                    ids1, u1 = call_func(
                        qs.query,
                        X=self.X,
                        y=y,
                        clf=clf,
                        X_eval=self.X,
                        ensemble=self.ensemble,
                        return_utilities=True,
                    )
                    ids2, u2 = call_func(
                        qs.query,
                        X=self.X,
                        y=y,
                        clf=clf,
                        X_eval=self.X,
                        candidates=unld_idx,
                        ensemble=self.ensemble,
                        return_utilities=True,
                    )
                    np.testing.assert_allclose(
                        u1[0][unld_idx], u2[0][unld_idx]
                    )

                try:
                    unld_idx = unlabeled_indices(y, self.MISSING_LABEL)
                    ids3, u3 = call_func(
                        qs.query,
                        X=self.X,
                        y=y,
                        clf=clf,
                        X_eval=self.X,
                        candidates=self.X[unld_idx],
                        ensemble=self.ensemble,
                        return_utilities=True,
                    )
                    np.testing.assert_array_equal(u1[0][unld_idx], u3[0])
                except MappingError:
                    pass

            for init_budget in [5, 1, 0]:
                y = np.full(self.y_true.shape, self.MISSING_LABEL)
                y[0:init_budget] = self.y_true[0:init_budget]

                with self.subTest(
                        msg="Basic AL Cycle",
                        init_budget=init_budget,
                        qs_name=qs_name,
                ):
                    qs = call_func(
                        self.query_strategies[qs_name],
                        only_mandatory=True,
                        missing_label=self.MISSING_LABEL,
                        classes=np.unique(self.y_true),
                        random_state=1,
                        ensemble=self.ensemble,
                    )

                    for b in range(self.budget):
                        q_id = call_func(
                            qs.query,
                            X=self.X,
                            y=y,
                            clf=clf,
                            X_eval=self.X,
                            ensemble=self.ensemble,
                        )
                        y[q_id] = self.y_true[q_id]

    def test_param(self):
        not_test = [
            "self",
            "kwargs",
            "missing_label",
            "random_state",
            "X",
            "y",
            "candidates",
            "batch_size",
            "return_utilities",
        ]
        for qs_name in self.query_strategies:
            clf = self.cmm if qs_name == "FourDs" else self.clf
            with self.subTest(msg="Param Test", qs_name=qs_name):
                # Get initial parameters.
                qs_class = self.query_strategies[qs_name]
                init_params = inspect.signature(qs_class).parameters.keys()
                init_params = list(init_params)

                # Get query parameters.
                query_params = inspect.signature(qs_class.query).parameters
                query_params = list(query_params.keys())

                # Check initial parameters.
                values = [Dummy() for i in range(len(init_params))]
                qs_obj = qs_class(*values)
                for param, value in zip(init_params, values):
                    self.assertTrue(
                        hasattr(qs_obj, param),
                        msg=f'"{param}" not tested for __init__()',
                    )
                    self.assertEqual(getattr(qs_obj, param), value)

                # Get class to check.
                class_filename = path.basename(inspect.getfile(qs_class))[:-3]
                mod = "skactiveml.pool.tests.test" + class_filename
                mod = import_module(mod)
                test_class_name = "Test" + qs_class.__name__
                msg = f"{qs_name} has no test called {test_class_name}."
                self.assertTrue(hasattr(mod, test_class_name), msg=msg)
                test_obj = getattr(mod, test_class_name)

                # Check init parameters.
                for param in np.setdiff1d(init_params, not_test):
                    test_func_name = "test_init_param_" + param
                    self.assertTrue(
                        hasattr(test_obj, test_func_name),
                        msg="'{}()' missing for parameter '{}' of "
                            "__init__()".format(test_func_name, param),
                    )

                # Check query parameters.
                for param in np.setdiff1d(query_params, not_test):
                    test_func_name = "test_query_param_" + param
                    msg = (
                        f"'{test_func_name}()' missing for parameter "
                        f"'{param}' of query()"
                    )
                    self.assertTrue(hasattr(test_obj, test_func_name), msg)

                # Check standard parameters of `__init__` method.
                self._test_init_param_random_state(qs_class, clf)
                self._test_init_param_missing_label(qs_class, clf)

                # Check standard parameters of `query` method.
                self._test_query_param_X(qs_class, clf)
                self._test_query_param_y(qs_class, clf)
                self._test_query_param_candidates(qs_class, clf)
                self._test_query_param_batch_size(qs_class, clf)
                self._test_query_param_return_utilities(qs_class, clf)

    def _test_init_param_random_state(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        self.assertTrue(qs_mdl.random_state is None)
        qs_mdl = call_func(
            qs_class,
            classes=np.unique(self.y_true),
            clf=clf,
            random_state="Test",
        )
        self.assertEqual(qs_mdl.random_state, "Test")
        self.assertRaises(
            ValueError,
            call_func,
            qs_mdl.query,
            clf=clf,
            X=self.X,
            y=self.y,
            ensemble=self.ensemble,
        )

    def _test_init_param_missing_label(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        check_equal_missing_label(qs_mdl.missing_label, MISSING_LABEL)
        qs_mdl = call_func(
            qs_class,
            classes=np.unique(self.y_true),
            clf=clf,
            missing_label=Dummy(),
        )
        self.assertRaises(
            TypeError,
            call_func,
            qs_mdl.query,
            clf=clf,
            X=self.X,
            y=self.y,
            ensemble=self.ensemble,
        )

    def _test_query_param_X(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        for X in [None, "str", [], np.ones(5)]:
            self.assertRaises(
                (TypeError, ValueError),
                call_func,
                qs_mdl.query,
                clf=clf,
                X=X,
                y=self.y,
                ensemble=self.ensemble,
            )

    def _test_query_param_y(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        for y in [None, "str", [], np.ones([5, 2])]:
            self.assertRaises(
                (TypeError, ValueError),
                call_func,
                qs_mdl.query,
                clf=clf,
                X=self.X,
                y=y,
                ensemble=self.ensemble,
            )
        self.assertRaises(
            ValueError,
            call_func,
            qs_mdl.query,
            clf=clf,
            X=self.X,
            y=self.y[:-2],
            ensemble=self.ensemble,
        )

    def _test_query_param_candidates(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        for candidates in [Dummy(), "test", 0]:
            self.assertRaises(
                (TypeError, ValueError),
                call_func,
                qs_mdl.query,
                candidates=candidates,
                clf=clf,
                X=self.X,
                y=self.y,
                ensemble=self.ensemble,
            )
        self.assertRaises(
            (TypeError, ValueError),
            call_func,
            qs_mdl.query,
            candidates=self.X[:, :1],
            clf=clf,
            X=self.X,
            y=self.y,
            ensemble=self.ensemble,
        )

    def _test_query_param_batch_size(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        self.assertRaises(
            ValueError,
            call_func,
            qs_mdl.query,
            clf=clf,
            X=self.X,
            y=self.y,
            batch_size=0,
            ensemble=self.ensemble,
        )
        self.assertRaises(
            TypeError,
            call_func,
            qs_mdl.query,
            clf=clf,
            X=self.X,
            y=self.y,
            batch_size=1.2,
            ensemble=self.ensemble,
        )

    def _test_query_param_return_utilities(self, qs_class, clf):
        qs_mdl = call_func(qs_class, classes=np.unique(self.y_true))
        self.assertRaises(
            TypeError,
            call_func,
            qs_mdl.query,
            clf=clf,
            X=self.X,
            y=self.y,
            return_utilities="test",
            ensemble=self.ensemble,
        )


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.skaml_path = path.abspath(os.curdir).split("skactiveml")[0]
        self.json_path = path.join(self.skaml_path, "docs", "examples", "pool")
        self.exceptions = []
        self.working_dir = os.curdir

    def test_pool_example_files(self):
        # Temporary generate the examples from the json files.
        examples_path = path.join(
            self.skaml_path, "docs", "temp_examples_pool"
        )
        generate_examples(examples_path, pool, self.json_path)

        # Execute the examples.
        pool_examples_path = path.join(examples_path, "examples", "pool")
        for filename in listdir(pool_examples_path):
            if filename.endswith(".py"):
                with self.subTest(msg=filename):
                    file_path = path.join(pool_examples_path, filename)
                    exec(open(file_path, "r").read(), locals())

        # Remove the created examples from disk.
        shutil.rmtree(examples_path)

    def test_json(self):
        # Collect all strategies for which an example exists
        strats_with_json = []
        for filename in listdir(self.json_path):
            if not filename.endswith(".json"):
                continue
            with open(path.join(self.json_path, filename)) as file:
                for example in json.load(file):
                    if example["class"] not in strats_with_json:
                        strats_with_json.append(example["class"])

        # Test if there is a json example for every AL-strategy.
        for item in pool.__all__:
            with self.subTest(msg="JSON Test", qs_name=item):
                item_missing = (
                        inspect.isclass(getattr(pool, item))
                        and item not in self.exceptions
                        and item not in strats_with_json
                )
                self.assertFalse(
                    item_missing,
                    f'No json example found for "{item}". Please '
                    f"add an example in\n"
                    f"{self.json_path}.\n"
                    f"For information how to create one, see the "
                    f"Developers Guide. If {item} is not an "
                    f'AL-strategy, add "{item}" to the '
                    f'"exceptions" list in this test class.',
                )


class Dummy:
    def __init__(self):
        pass
