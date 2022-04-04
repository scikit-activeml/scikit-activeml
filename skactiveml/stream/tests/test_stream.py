import inspect
import unittest
from collections import deque
from importlib import import_module
from os import path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from skactiveml import stream
from skactiveml.base import SingleAnnotatorStreamQueryStrategy
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.utils import call_func


class TestStream(unittest.TestCase):
    def setUp(self):
        self.query_strategies = {}
        for qs_name in stream.__all__:
            qs = getattr(stream, qs_name)
            if inspect.isclass(qs) and issubclass(
                    qs, SingleAnnotatorStreamQueryStrategy
            ):
                self.query_strategies[qs_name] = qs
        self.clf = ParzenWindowClassifier()

    def test_selection_strategies(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        stream_length = 100
        train_init_size = 10
        training_size = 50
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )

        clf = ParzenWindowClassifier(
            classes=[0, 1], random_state=rand.randint(2 ** 31 - 1)
        )

        X_init = X[:train_init_size, :]
        y_init = y[:train_init_size]

        X_stream = X[train_init_size:, :]
        y_stream = y[train_init_size:]

        # # Build dictionary of attributes.
        # query_strategy_classes = {}
        # for s_class in stream.__all__:
        #     query_strategy_classes[s_class] = getattr(stream, s_class)

        # Test predictions of classifiers.
        for qs_name, qs_class in self.query_strategies.items():
            self._test_query_strategy(
                rand.randint(2 ** 31 - 1),
                qs_class,
                clf,
                X_init,
                y_init,
                X_stream,
                y_stream,
                training_size,
                qs_name,
            )
            self._test_update_before_query(
                rand.randint(2 ** 31 - 1),
                qs_class,
                clf,
                X_init,
                y_init,
                X_stream,
                y_stream,
                training_size,
                qs_name,
            )

    def _test_query_strategy(
            self,
            rand_seed,
            query_strategy_class,
            clf,
            X_init,
            y_init,
            X_stream,
            y_stream,
            training_size,
            qs_name,
    ):

        rand = check_random_state(rand_seed)
        random_state = rand.randint(2 ** 31 - 1)
        query_strategy = query_strategy_class(random_state=random_state)

        query_strategy2 = query_strategy_class(random_state=random_state)

        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)

        for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
            return_utilities = t % 2 == 0
            qs_output = call_func(
                query_strategy.query,
                candidates=x_t.reshape([1, -1]),
                clf=clf,
                return_utilities=return_utilities,
            )

            for i in range(3):
                qs_output2 = call_func(
                    query_strategy2.query,
                    candidates=x_t.reshape([1, -1]),
                    clf=clf,
                    return_utilities=return_utilities,
                )

            if return_utilities:
                queried_indices, utilities = qs_output
                queried_indices2, utilities2 = qs_output2
                self.assertEqual(utilities, utilities2)
            else:
                queried_indices = qs_output
                queried_indices2 = qs_output2
                utilities = [0.5]
                utilities2 = [0.5]
            self.assertEqual(len(queried_indices), len(queried_indices2))
            budget_manager_param_dict1 = {"utilities": utilities}
            budget_manager_param_dict2 = {"utilities": utilities2}
            call_func(
                query_strategy.update,
                candidates=x_t.reshape([1, -1]),
                queried_indices=queried_indices,
                budget_manager_param_dict=budget_manager_param_dict1,
            )
            call_func(
                query_strategy2.update,
                candidates=x_t.reshape([1, -1]),
                queried_indices=queried_indices2,
                budget_manager_param_dict=budget_manager_param_dict2,
            )
            X_train.append(x_t)
            if len(queried_indices):
                y_train.append(y_t)
            else:
                y_train.append(clf.missing_label)
            clf.fit(X_train, y_train)

    def _test_update_before_query(
            self,
            rand_seed,
            query_strategy_class,
            clf,
            X_init,
            y_init,
            X_stream,
            y_stream,
            training_size,
            qs_name,
    ):
        rand = check_random_state(rand_seed)
        random_state = rand.randint(2 ** 31 - 1)
        query_strategy = query_strategy_class(random_state=random_state)

        query_strategy2 = query_strategy_class(random_state=random_state)

        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)

        for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
            return_utilities = t % 2 == 0
            qs_output = call_func(
                query_strategy.query,
                candidates=x_t.reshape([1, -1]),
                clf=clf,
                return_utilities=return_utilities,
            )

            if return_utilities:
                queried_indices, utilities = qs_output
            else:
                queried_indices = qs_output
                utilities = [0.5]
            budget_manager_param_dict1 = {"utilities": utilities}
            budget_manager_param_dict2 = {"utilities": utilities}
            call_func(
                query_strategy.update,
                candidates=x_t.reshape([1, -1]),
                queried_indices=queried_indices,
                budget_manager_param_dict=budget_manager_param_dict1,
            )
            call_func(
                query_strategy2.update,
                candidates=x_t.reshape([1, -1]),
                queried_indices=queried_indices,
                budget_manager_param_dict=budget_manager_param_dict2,
            )
            X_train.append(x_t)
            if len(queried_indices):
                y_train.append(y_t)
            else:
                y_train.append(clf.missing_label)
            clf.fit(X_train, y_train)

    def test_param(self):
        not_test = ["self", "kwargs"]
        for qs_name in self.query_strategies:
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
                mod = "skactiveml.stream.tests.test" + class_filename
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


class Dummy:
    def __init__(self):
        pass
