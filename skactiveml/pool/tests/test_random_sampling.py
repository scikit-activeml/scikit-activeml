import unittest

import numpy as np
from numpy.random import RandomState

from skactiveml.pool import RandomSampling
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestRandomSampling(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_clf = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.hstack([[0, 1], np.full(8, MISSING_LABEL)]),
        }
        qs_params_clf_multilabel = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.vstack(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                    np.full(2, MISSING_LABEL, dtype=float),
                ]
            ),
        }
        query_default_params_reg = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.hstack([[1.1, 2.1], np.full(8, MISSING_LABEL)]),
        }
        super().setUp(
            qs_class=RandomSampling,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
            query_default_params_clf_multilabel=qs_params_clf_multilabel,
        )

    def test_query(self):
        rand1 = RandomSampling(random_state=RandomState(14))
        rand2 = RandomSampling(random_state=14)

        X = np.zeros([10, 2])
        y = np.hstack([[0, 1], np.full(8, MISSING_LABEL)])

        self.assertEqual(rand1.query(X, y), rand1.query(X, y))
        self.assertEqual(rand1.query(X, y), rand2.query(X, y))

        qidx = rand1.query(X, y)
        self.assertEqual(len(qidx), 1)
