import unittest
import numpy as np
from collections import deque

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from skactiveml import stream
from skactiveml.utils import call_func
from skactiveml.classifier import PWC


class TestStream(unittest.TestCase):
    def test_selection_strategies(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        stream_length = 1000
        train_init_size = 10
        training_size = 100
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )

        clf = PWC(classes=[0, 1], random_state=rand.randint(2 ** 31 - 1))

        X_init = X[:train_init_size, :]
        y_init = y[:train_init_size]

        X_stream = X[train_init_size:, :]
        y_stream = y[train_init_size:]

        # Build dictionary of attributes.
        query_strategy_classes = {}
        for s_class in stream.__all__:
            query_strategy_classes[s_class] = getattr(stream, s_class)

        # Test predictions of classifiers.
        for qs_name, qs_class in query_strategy_classes.items():
            self._test_selection_strategy(
                rand.randint(2 ** 31 - 1),
                qs_class,
                clf,
                X_init,
                y_init,
                X_stream,
                y_stream,
                training_size,
                qs_name
            )

    def _test_selection_strategy(
        self,
        rand_seed,
        query_strategy_class,
        clf,
        X_init,
        y_init,
        X_stream,
        y_stream,
        training_size,
        qs_name
    ):

        rand = check_random_state(rand_seed)
        random_state = rand.randint(2 ** 31 - 1)
        query_strategy = query_strategy_class(
            random_state=random_state
        )

        query_strategy2 = query_strategy_class(
            random_state=random_state
        )

        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)

        for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
            return_utilities = t % 2 == 0
            qs_output = call_func(query_strategy.query,
                                  X_cand=x_t.reshape([1, -1]),
                                  clf=clf,
                                  return_utilities=return_utilities
                                  )

            for i in range(3):
                qs_output2 = call_func(query_strategy2.query,
                                       X_cand=x_t.reshape([1, -1]),
                                       clf=clf,
                                       return_utilities=return_utilities
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
            query_strategy.update(
                x_t.reshape([1, -1]),
                queried_indices,
                budget_manager_param_dict1
            )
            query_strategy2.update(
                x_t.reshape([1, -1]),
                queried_indices,
                budget_manager_param_dict2
            )
            X_train.append(x_t)
            if len(queried_indices):
                y_train.append(y_t)
            else:
                y_train.append(clf.missing_label)
            clf.fit(X_train, y_train)
