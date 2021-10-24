import unittest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from skactiveml import stream
from skactiveml.utils import call_func
from skactiveml.classifier import PWC
from collections import deque


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
            if qs_name == "PALS":
                return_utilities = True
            qs_output = call_func(query_strategy.query,
                                  X_cand=x_t.reshape([1, -1]),
                                  clf=clf,
                                  return_utilities=return_utilities
                                  )

            for _ in range(3):
                qs_output2 = call_func(query_strategy2.query,
                                       X_cand=x_t.reshape([1, -1]),
                                       clf=clf,
                                       return_utilities=return_utilities
                                       )

            if return_utilities:
                sampled_indices, utilities = qs_output
                sampled_indices2, utilities2 = qs_output2
                self.assertEqual(utilities, utilities2)
            else:
                sampled_indices = qs_output
                sampled_indices2 = qs_output2
                utilities = None
                utilities2 = None
            self.assertEqual(len(sampled_indices), len(sampled_indices2))
            query_strategy.update(
                x_t.reshape([1, -1]), sampled_indices, utilities=utilities
            )
            query_strategy2.update(
                x_t.reshape([1, -1]), sampled_indices, utilities=utilities2
            )
            if len(sampled_indices):
                # X_train.append(x_t)
                # y_train.append(y_t)
                clf.fit(X_train, y_train)
