import inspect
import unittest
import warnings
from copy import deepcopy

import numpy as np
from numpy.random import RandomState

from skactiveml.exceptions import MappingError
from skactiveml.utils import MISSING_LABEL, is_unlabeled, is_labeled, \
    unlabeled_indices

from sklearn.naive_bayes import GaussianNB

class Dummy:
    def __init__(self):
        pass

class TemplateQueryStrategy:

    def setUp(self, qs_class, init_default_params,
              query_default_params_clf=None, query_default_params_reg=None):
        self.super_setUp_has_been_executed = True
        self.qs_class = qs_class

        self.init_default_params = {"random_state": 42}
        for key, val in init_default_params.items():
            self.init_default_params[key] = val

        init_params = inspect.signature(self.qs_class.__init__).parameters
        for key, val in init_params.items():
            if key != "self" and val.default == inspect._empty and\
                    key not in self.init_default_params:
                raise ValueError(f"Missing positional argument `{key}` of "
                                 f"`__init__` in "
                                 f"`init_default_kwargs`.")

        self.query_default_params_clf = query_default_params_clf
        self.query_default_params_reg = query_default_params_reg

        if self.query_default_params_clf is None and \
                self.query_default_params_reg is None:
            raise ValueError(f"The query strategies must support either "
                             f"classification or regression. Hence, at least "
                             f"one parameter of `query_default_params_clf` "
                             f"and `query_default_params_reg` must be not None. "
                             f"Use emtpy dictionary to use default values.")
        query_params = inspect.signature(self.qs_class.query).parameters
        for key, val in query_params.items():
            if key != "self" and val.default == inspect._empty and \
                    self.query_default_params_clf is not None and \
                    key not in self.query_default_params_clf:
                raise ValueError(f"Missing positional argument `{key}` of "
                                 f"`query` in "
                                 f"`query_default_kwargs_clf`.")

            if key != "self" and val.default == inspect._empty and \
                    self.query_default_params_reg is not None and \
                    key not in self.query_default_params_reg:
                raise ValueError(f"Missing positional argument `{key}` of "
                                 f"`query` in "
                                 f"`query_default_kwargs_reg`.")


    def test_init_param_random_state(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), (1, None)]
        self._test_param("init", "random_state", test_cases)

    def test_query_param_fit_clf(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.query).parameters
        if "fit_clf" in query_params:
            # custom test cases are not necessary
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, TypeError), ("state", TypeError)]
            self._test_param("query", "fit_clf", test_cases)

            # check if clf remains the same for both options
            for fit_clf in [True, False]:
                with self.subTest(msg="Clf consistency", fit_clf=fit_clf):
                    clf = deepcopy(self.query_default_params_clf["clf"])
                    if not fit_clf:
                        clf.fit(self.query_default_params_clf["X"],
                                self.query_default_params_clf["y"])
                    query_params = deepcopy(self.query_default_params_clf)
                    query_params["clf"] = deepcopy(clf)
                    query_params["fit_clf"] = fit_clf

                    qs = self.qs_class(**self.init_default_params)
                    qs.query(**query_params)
                    self.assertTrue(
                        _cmp_object_dict(
                            query_params["clf"].__dict__,
                            clf.__dict__
                        ),
                        msg=f"Classifier changed after calling query for "
                            f"`fit_clf={fit_clf}`."
                    )

    def test_query_param_clf(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.query).parameters
        if "clf" in query_params:
            # custom test cases are necessary as clf usually has specific
            # properties for query strategies
            if test_cases is None:
                raise NotImplementedError(
                    "The test function `test_query_param_clf` should be "
                    "implemented for every query strategy as they probably "
                    "have specific demands. If the query strategy supports "
                    "every classifier, please call "
                    "`super().test_query_param_clf(test_cases=[])`."
                )
            test_cases += [(np.nan, TypeError), ("state", TypeError),
                           (Dummy(), TypeError), (GaussianNB(), TypeError)]
            self._test_param("query", "clf", test_cases)

            # check if clf remains the same
            with self.subTest(msg="Clf consistency"):
                clf = deepcopy(self.query_default_params_clf["clf"])
                query_params = deepcopy(self.query_default_params_clf)
                query_params["clf"] = deepcopy(clf)

                qs = self.qs_class(**self.init_default_params)
                qs.query(**query_params)
                self.assertTrue(
                    _cmp_object_dict(
                        query_params["clf"].__dict__,
                        clf.__dict__
                    ),
                    msg=f"Classifier changed after calling query."
                )

    def _test_param(self, test_func, test_param, test_cases,
                    replace_init_params=None, replace_query_params=None,
                    exclude_clf=False, exclude_reg=False):
        if replace_init_params is None:
            replace_init_params = {}
        if replace_query_params is None:
            replace_query_params = {}

        for i, (test_val, err) in enumerate(test_cases):
            # TODO Subtest to show which test failed
            with self.subTest(msg="Param", id=i, val=test_val):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                for query_params, exclude_case in \
                        [(self.query_default_params_clf, exclude_clf),
                         (self.query_default_params_reg, exclude_reg)]:
                    if not (query_params is None or exclude_case):
                        query_params = deepcopy(query_params)
                        for key, val in replace_query_params.items():
                            query_params[key] = val

                        locals()[f"{test_func}_params"][test_param] = test_val

                        qs = self.qs_class(**init_params)
                        if err is None:
                            qs.query(**query_params)
                        else:
                            self.assertRaises(err, qs.query, **query_params)

    def test_init_param_assignments(self):
        for param in inspect.signature(self.qs_class.__init__).parameters:
            if param != "self":
                init_params = deepcopy(self.init_default_params)
                init_params[param] = Dummy()
                qs = self.qs_class(**init_params)
                self.assertEqual(
                    getattr(qs, param), init_params[param],
                    msg=f"The parameter `{param}` was not assigned to a class "
                        f"variable when `__init__` was called."
                )

    def test_param_test_availability(self):
        not_test = ["self", "kwargs"]

        # Get initial parameters.
        init_params = inspect.signature(self.qs_class.__init__).parameters
        init_params = list(init_params.keys())

        # Check init parameters.
        for param in np.setdiff1d(init_params, not_test):
            test_func_name = "test_init_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}"
                )

        # Get query parameters.
        query_params = inspect.signature(self.qs_class.query).parameters
        query_params = list(query_params.keys())

        # Check init parameters.
        for param in np.setdiff1d(query_params, not_test):
            test_func_name = "test_query_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}"
                )

        # Check if query is being tested.
        with self.subTest(msg="test_query"):
            self.assertTrue(
                hasattr(self, "test_query"),
                msg=f"'test_query' missing in {self.__class__}"
            )

class TemplatePoolQueryStrategy(TemplateQueryStrategy):

    def setUp(self, qs_class, init_default_params,
              query_default_params_clf=None, query_default_params_reg=None):
        if 'missing_label' not in init_default_params:
            init_default_params['missing_label'] = MISSING_LABEL
        super().setUp(qs_class, init_default_params,
                      query_default_params_clf, query_default_params_reg)
        self.y_shape = list(self.query_default_params_clf["y"].shape \
                            if self.query_default_params_clf is not None else \
                            self.query_default_params_reg["y"].shape)

    def test_init_param_missing_label(self, test_cases=None):  # TODO add more cases
        test_cases = [] if test_cases is None else test_cases
        self._test_param("init", "missing_label", test_cases)

        # Todo replace missing value in `y` as well
        if self.query_default_params_clf is not None:
            test_cases = [(np.nan, None), (Dummy, TypeError)]
            self._test_param("init", "missing_label", test_cases, exclude_reg=True)
        if self.query_default_params_reg is not None:
            test_cases = [(1, ValueError), ("string", TypeError), (Dummy, TypeError)]
            self._test_param("init", "missing_label", test_cases, exclude_clf=True)

    def test_query_param_X(self, test_cases=None):  #TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", (ValueError, TypeError)),
                       (Dummy, (ValueError, TypeError))]
        replace_init_params = {"missing_label": MISSING_LABEL}
        replace_query_params = {"y": np.full(self.y_shape, MISSING_LABEL),
                                "candidates": None}
        test_cases += [(np.zeros([self.y_shape[0], 3]), None),
                       (np.zeros([self.y_shape[0]+1, 3]), ValueError)]
        self._test_param("query", "X", test_cases,
                         replace_init_params=replace_init_params,
                         replace_query_params=replace_query_params)

    def test_query_param_y(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, TypeError), (Dummy, TypeError)]
        self._test_param("query", "y", test_cases)

        if self.query_default_params_clf is not None:
            y = self.query_default_params_clf["y"]
            test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
            self._test_param("query", "y", test_cases, exclude_reg=True)
        if self.query_default_params_reg is not None:
            y = self.query_default_params_reg["y"]
            test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
            self._test_param("query", "y", test_cases, exclude_clf=True)

    def test_query_param_candidates(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), (Dummy, TypeError),
                       ([0], None)]
        self._test_param("query", "candidates", test_cases)

    def test_query_param_sample_weight(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.query).parameters
        if "sample_weight" in query_params:
            # custom test cases are not necessary
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, (ValueError,TypeError)),
                           (Dummy, (ValueError,TypeError)),
                           (None, None)]
            self._test_param("query", "sample_weight", test_cases)

            for exclude_clf, exclude_reg, query_params in \
                    [(True, False, self.query_default_params_clf),
                     (False, True, self.query_default_params_reg)]:
                if self.query_default_params_reg is not None:
                    y = query_params["y"]
                    test_cases = [(np.ones(len(y)), None),
                                  (np.ones(len(y)+1), ValueError)]
                    self._test_param("query", "sample_weight", test_cases,
                                     exclude_clf=exclude_clf, exclude_reg=exclude_reg)

    def test_query_param_utility_weight(self,
                                       test_cases=None):  # TODO more cases

        query_params = inspect.signature(self.qs_class.query).parameters
        if "utility_weight" in query_params:
            # custom test cases are not necessary
            raise NotImplementedError("TODO Daniel")

    def test_query_param_batch_size(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(0, ValueError), (1.2, TypeError), (1, None)]
        self._test_param("query", "batch_size", test_cases)

    def test_query_param_return_utilities(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", TypeError), (Dummy, TypeError), (True, None)]
        self._test_param("query", "return_utilities", test_cases)

    def test_query_reproducibility(self):
        # checks if the results stays the same with same random state
        init_params = deepcopy(self.init_default_params)
        init_params["random_state"] = np.random.RandomState(0)

        qs = self.qs_class(**init_params)

        for query_params in [self.query_default_params_clf,
                               self.query_default_params_reg]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                query_params["return_utilities"] = True
                id1, u1 = qs.query(**query_params)
                id2, u2 = qs.query(**query_params)

                self.assertEqual(len(u1[0]), len(query_params["X"]))
                np.testing.assert_array_equal(id1, id2)
                np.testing.assert_allclose(u1, u2)

class TemplateSingleAnnotatorPoolQueryStrategy(TemplatePoolQueryStrategy):

    def test_query_al_cycles(self):
        budget = 1
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)

        for query_params in [self.query_default_params_clf,
                             self.query_default_params_reg]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                
                missing_label = self.init_default_params['missing_label']
                lbld_idx = is_labeled(query_params["y"], missing_label)
                unld_idx = is_unlabeled(query_params["y"], missing_label)
                y_true = deepcopy(query_params["y"])
                y_true[unld_idx] = y_true[lbld_idx][0]

                for init_labels in [0, 1, sum(unld_idx)-1]:
                    y = np.full(y_true.shape, fill_value=missing_label)
                    y[0:init_labels] = y_true[0:init_labels]
                    
                    with self.subTest(init_labels=init_labels):
                        for b in range(budget):
                            query_params["y"] = y
                            query_id = qs.query(**query_params)
                            query_params["y"][query_id] = y_true[query_id]

    def test_query_batch_variation(self):
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)

        for query_params in [self.query_default_params_clf,
                             self.query_default_params_reg]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                missing_label = self.init_default_params['missing_label']
                max_batch_size = \
                    int(sum(is_unlabeled(query_params["y"], missing_label)))
                batch_size = min(5, max_batch_size)
                self.assertTrue(batch_size > 1, msg="Too few unlabeled")

                query_params["batch_size"] = batch_size
                query_params["return_utilities"] = True
                query_ids, utils = qs.query(**query_params)

                self.assertEqual(len(query_ids), batch_size)
                self.assertEqual(len(utils), batch_size)
                self.assertEqual(len(utils[0]), len(query_params["X"]))
                n_labeled = sum(is_labeled(query_params["y"], missing_label))
                self.assertEqual(sum(np.isnan(utils[0])), n_labeled)

                query_params["batch_size"] = max_batch_size + 1
                query_params["return_utilities"] = False
                self.assertWarns(Warning, qs.query, **query_params)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    ids = qs.query(**query_params)
                    self.assertEqual(len(ids), max_batch_size)

    def test_query_candidate_variation(self):
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)
        missing_label = self.init_default_params['missing_label']

        for query_params in [self.query_default_params_clf,
                             self.query_default_params_reg]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                query_params["candidates"] = None
                query_params["return_utilities"] = True

                query_idx1, utils1 = qs.query(**query_params)

                unld_idx = unlabeled_indices(query_params["y"], missing_label)
                query_params["candidates"] = unld_idx
                query_idx2, utils2 = qs.query(**query_params)

                unld_idx2 = unld_idx[0:1]
                query_params["candidates"] = unld_idx2
                query_idx3, utils3 = qs.query(**query_params)

                np.testing.assert_allclose(utils1, utils2)
                utils3_copy = np.full_like(utils1, fill_value=np.nan)
                utils3_copy[0, unld_idx2] = utils3[0, unld_idx2]
                np.testing.assert_allclose(utils3, utils3_copy)

                try:
                    query_params["candidates"] = query_params["X"][unld_idx]
                    query_idx4, utils4 = qs.query(**query_params)

                    np.testing.assert_allclose(utils1[0][unld_idx], utils4[0])
                except MappingError:
                    pass

def _cmp_object_dict(d1, d2):
    keys = np.union1d(d1.keys(), d2.keys())[0]
    print(keys)
    for key in keys:
        print(f"{key}..")
        if key not in d1.keys() or key not in d2.keys():
            return False
        if hasattr(d1[key], "__dict__") ^ hasattr(d1[key], "__dict__"):
            return False
        if hasattr(d1[key], "__dict__") and hasattr(d1[key], "__dict__"):
            print("  .. go into")
            if not _cmp_object_dict(d1[key].__dict__, d2[key].__dict__):
                return False
            print("  .. go back")
        try:
            if np.issubdtype(type(d1[key]), np.number) and np.issubdtype(type(d1[key]), np.number):
                if np.isnan(d1[key]) == np.isnan(d2[key]):
                    pass
                elif np.isnan(d1[key]) ^ np.isnan(d2[key]):
                    return False
                else:
                    if not d1[key].__eq__(d2[key]):
                        return False
        except NotImplementedError:
            pass
        except Exception:
            return False
        print(f"  .. passed")
    return True
