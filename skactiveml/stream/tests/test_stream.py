import unittest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from skactiveml import stream
from skactiveml.classifier import PWC
from collections import deque


class TestStream(unittest.TestCase):
    def test_selection_strategies(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        stream_length = 3000
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
    ):
        rand = check_random_state(rand_seed)
        query_strategy = query_strategy_class(
            random_state=rand.randint(2 ** 31 - 1)
        )

        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)

        for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
            return_utilities = t % 2 == 0
            qs_output = query_strategy.query(
                x_t.reshape([1, -1]),
                clf=clf,
                X=X_train,
                y=y_train,
                sample_weight=np.ones(len(y_train)),
                return_utilities=return_utilities
            )
            if return_utilities:
                sampled_indices, utilities = qs_output
            else:
                sampled_indices = qs_output

            query_strategy.update(
                x_t.reshape([1, -1]), sampled_indices
            )
            if len(sampled_indices):
                X_train.append(x_t)
                y_train.append(y_t)
                # clf.fit(X_train, y_train)
