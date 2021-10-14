import unittest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from skactiveml.stream import verification_latency
from skactiveml.classifier import PWC
from collections import deque


class TestStreamDelay(unittest.TestCase):
    def test_selection_strategies(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        stream_length = 300
        train_init_size = 10
        training_size = 100
        verification_latencies = 60
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

        tX = np.arange(stream_length)
        ty = tX + verification_latencies
        tX_init = tX[:train_init_size]
        ty_init = ty[:train_init_size]
        tX_stream = tX[train_init_size:]
        ty_stream = ty[train_init_size:]

        # Build dictionary of attributes.
        query_strategy_classes = {}
        for s_class in verification_latency.__all__:
            query_strategy_classes[s_class] = getattr(
                verification_latency, s_class
            )

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
                tX_init,
                ty_init,
                tX_stream,
                ty_stream,
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
        tX_init,
        ty_init,
        tX_stream,
        ty_stream,
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
        tX_train = deque(maxlen=training_size)
        tX_train.extend(tX_init)
        ty_train = deque(maxlen=training_size)
        ty_train.extend(ty_init)
        acquisitions = deque(maxlen=training_size)
        acquisitions.extend(np.full(len(y_train), True))

        for t, (x_t, y_t, tX_t, ty_t) in enumerate(
            zip(X_stream, y_stream, tX_stream, ty_stream)
        ):
            queried_indices = query_strategy.query(
                X_cand=x_t.reshape([1, -1]),
                clf=clf,
                X=X_train,
                y=y_train,
                tX=tX_train,
                ty=ty_train,
                tX_cand=np.array([tX_t]),
                ty_cand=np.array([ty_t]),
                acquisitions=acquisitions,
            )
            query_strategy.update(
                x_t.reshape([1, -1]), queried_indices, X=X_train, y=y_train
            )

            acquisitions.append((len(queried_indices) > 0))
            # add the current instance to the training data
            tX_train.append(tX_t)
            ty_train.append(ty_t)
            X_train.append(x_t)
            y_train.append(y_t)
            # clf.fit(X_train, y_train)
