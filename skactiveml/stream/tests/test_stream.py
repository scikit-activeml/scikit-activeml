import unittest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from skactiveml import stream
from collections import deque


class TestStream(unittest.TestCase):
    def test_selection_strategies(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        stream_length = 300
        train_init_size = 10
        training_size = 100
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )
        tX = np.array(range(len(X)))
        ty = tX + 20

        tX_init = tX[:train_init_size]
        ty_init = ty[:train_init_size]

        X_init = X[:train_init_size, :]
        y_init = y[:train_init_size]

        tX_stream = tX[train_init_size:]
        ty_stream = ty[train_init_size:]

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
                X_init,
                y_init,
                X_stream,
                y_stream,
                training_size,
                tX_init,
                ty_init,
                tX_stream,
                ty_stream,
            )

    def _test_selection_strategy(
        self,
        rand_seed,
        query_strategy_class,
        X_init,
        y_init,
        X_stream,
        y_stream,
        training_size,
        tX_init,
        ty_init,
        tX_stream,
        ty_stream,
    ):
        rand = check_random_state(rand_seed)
        query_strategy = query_strategy_class(
            random_state=rand.randint(2 ** 31 - 1)
        )

        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)
        acquisitions = deque(maxlen=training_size)
        acquisitions.extend(np.full(len(y_train),True))
        tX_train = deque(maxlen=training_size)
        tX_train.extend(tX_init)
        ty_train = deque(maxlen=training_size)
        ty_train.extend(ty_init)

        for t, (X_cand, y_cand, tX_cand, ty_cand) in enumerate(zip(X_stream, y_stream, tX_stream, ty_stream)):
            sampled_indices = query_strategy.query(
                X_cand.reshape([1, -1]), X=X_train, y=y_train, tX=tX_train,
                tX_cand=np.array([tX_cand]),
                ty=ty_train,
                ty_cand=np.array([ty_cand]),
                acquisitions=acquisitions,
            )
            if len(sampled_indices):
                X_train.append(X_cand)
                y_train.append(y_cand)
                tX_train.append(tX_cand)
                ty_train.append(ty_cand)
                acquisitions.append(True)
                # clf.fit(X_train, y_train)

    def test_query_update(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        stream_length = 300
        train_init_size = 10
        training_size = 100
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )
        tX = np.array(range(len(X)))
        ty = tX + 20

        tX_init = tX[:train_init_size]
        ty_init = ty[:train_init_size]

        X_init = X[:train_init_size, :]
        y_init = y[:train_init_size]

        tX_stream = tX[train_init_size:]
        ty_stream = ty[train_init_size:]

        X_stream = X[train_init_size:, :]
        y_stream = y[train_init_size:]

        # Build dictionary of attributes.
        query_strategy_classes = {}
        for s_class in stream.__all__:
            query_strategy_classes[s_class] = getattr(stream, s_class)
        # Test predictions of classifiers.
        for qs_name, qs_class in query_strategy_classes.items():
            self._test_query_update(
                rand.randint(2 ** 31 - 1),
                qs_class,
                X_init,
                y_init,
                X_stream,
                y_stream,
                training_size,
                tX_init,
                ty_init,
                tX_stream,
                ty_stream,
            )

    def _test_query_update(
        self,
        rand_seed,
        query_strategy_class,
        X_init,
        y_init,
        X_stream,
        y_stream,
        training_size,
        tX_init,
        ty_init,
        tX_stream,
        ty_stream,
    ):
        rand = check_random_state(rand_seed)
        qs_rand_seed = rand.randint(2 ** 31 - 1)
        query_strategy_1 = query_strategy_class(random_state=qs_rand_seed)
        query_strategy_2 = query_strategy_class(random_state=qs_rand_seed)
        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)
        acquisitions = deque(maxlen=training_size)
        acquisitions.extend(np.full(len(y_train),True))
        tX_train = deque(maxlen=training_size)
        tX_train.extend(tX_init)
        ty_train = deque(maxlen=training_size)
        ty_train.extend(ty_init)

        for t, (X_cand, y_cand, tX_cand, ty_cand) in enumerate(zip(X_stream, y_stream, tX_stream, ty_stream)):
            sampled_indices_1, utilities_1 = query_strategy_1.query(
                X_cand.reshape([1, -1]),
                X=X_train,
                y=y_train,
                tX=tX_train,
                tX_cand=np.array([tX_cand]),
                ty=ty_train,
                ty_cand=np.array([ty_cand]),
                acquisitions=acquisitions,
                return_utilities=True,
            )

            sampled_indices_2, utilities_2 = query_strategy_2.query(
                X_cand.reshape([1, -1]),
                X=X_train,
                y=y_train,
                tX=tX_train,
                tX_cand=np.array([tX_cand]),
                ty=ty_train,
                ty_cand=np.array([ty_cand]),
                acquisitions=acquisitions,
                simulate=True,
                return_utilities=True,
            )
            sampled = np.array([len(sampled_indices_2) > 0])
            query_strategy_2.update(
                X_cand.reshape([1, -1]), sampled, X=X_train, y=y_train
            )

            if ((len(sampled_indices_1) != len(sampled_indices_2))
               or (utilities_1[0] != utilities_2[0])):
                print("query_strategy_class", query_strategy_class)
                print("t", t)

                print("sampled_indices_1", sampled_indices_1)
                print("utilities_1", utilities_1)

                print("sampled_indices_2", sampled_indices_2)
                print("utilities_2", utilities_2)
            self.assertEqual(utilities_1[0], utilities_2[0])
            self.assertEqual(len(sampled_indices_1), len(sampled_indices_2))

            if len(sampled_indices_1):
                X_train.append(X_cand)
                y_train.append(y_cand)
                tX_train.append(tX_cand)
                ty_train.append(ty_cand)
                acquisitions.append(True)
