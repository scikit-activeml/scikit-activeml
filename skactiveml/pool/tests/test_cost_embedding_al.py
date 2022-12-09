import unittest

import numpy as np
from sklearn.svm import SVR

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import CostEmbeddingAL
from skactiveml.pool._cost_embedding_al import MDSP, smacof_p
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestCostEmbeddingAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.classes = [0, 1]
        init_default_params = {"classes": self.classes}
        query_default_params = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.hstack([[0, 1], np.full(8, MISSING_LABEL)]),
        }
        super().setUp(
            qs_class=CostEmbeddingAL,
            init_default_params=init_default_params,
            query_default_params_clf=query_default_params,
        )

    # Test init parameters
    def test_init_param_classes(self):
        test_cases = [
            (True, TypeError),
            ("string", TypeError),
            (np.zeros(2), ValueError),
        ]
        self._test_param("init", "classes", test_cases)

    def test_init_param_base_regressor(self):
        test_cases = [
            (1, TypeError),
            ("string", TypeError),
            (ParzenWindowClassifier(), TypeError),
            (SVR(), None),
        ]
        self._test_param("init", "base_regressor", test_cases)

    def test_init_param_cost_matrix(self):
        test_cases = [
            (np.ones((len(self.classes), len(self.classes) + 1)), ValueError),
            ("string", ValueError),
            (np.ones((3, 3)), ValueError),
            (np.zeros((len(self.classes), len(self.classes))), ValueError),
            (
                np.ones((len(self.classes), len(self.classes)))
                - np.eye(len(self.classes)),
                None,
            ),
        ]
        self._test_param("init", "cost_matrix", test_cases)

    def test_init_param_embed_dim(self):
        test_cases = [
            (True, TypeError),
            ("string", TypeError),
            (1.5, TypeError),
            (0, ValueError),
            (3, None),
        ]
        self._test_param("init", "embed_dim", test_cases)

    def test_init_param_mds_params(self):
        test_cases = [
            (True, TypeError),
            ("string", TypeError),
            (0, TypeError),
            ({}, None),
        ]
        self._test_param("init", "mds_params", test_cases)

    def test_init_param_nn_params(self):
        test_cases = [
            (True, TypeError),
            ("string", TypeError),
            (0, TypeError),
            ({}, None),
        ]
        self._test_param("init", "nn_params", test_cases)

    # Test query
    def test_query(self):
        alce = CostEmbeddingAL(base_regressor=SVR(), classes=[0, 1])
        query_indices = alce.query(
            [[0], [200]], [0, 1], candidates=[[0], [100], [200]]
        )
        np.testing.assert_array_equal(query_indices, [1])

    def test_mds_params(self):
        np.random.seed(14)
        X = np.random.random((10, 2))
        y = np.random.randint(0, 2, 10)
        candidates = np.random.random((15, 2))
        classes = [0, 1, 2]
        cost_matrix = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
        regressor = SVR()

        alce = CostEmbeddingAL(
            classes,
            regressor,
            cost_matrix,
            random_state=14,
            mds_params={"n_jobs": 1, "verbose": 2},
        )
        cand1 = alce.query(X, y, candidates=candidates)
        alce = CostEmbeddingAL(
            classes,
            regressor,
            cost_matrix,
            random_state=14,
            mds_params={"n_jobs": 2},
        )
        cand2 = alce.query(X, y, candidates=candidates)
        np.testing.assert_array_equal(cand1, cand2)

        alce = CostEmbeddingAL(
            classes,
            regressor,
            cost_matrix,
            mds_params={"dissimilarity": "wrong"},
        )
        self.assertRaises(ValueError, alce.query, X, y, candidates=candidates)

        alce = CostEmbeddingAL(
            base_regressor=regressor,
            classes=[0, 1],
            mds_params={"dissimilarity": "precomputed"},
        )
        query_indices = alce.query(
            [[0], [200]], [0, 1], candidates=[[0], [100], [200]]
        )
        np.testing.assert_array_equal(query_indices, [1])

    def test_MDS(self):
        sim = np.array(
            [[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]]
        )
        mds_clf = MDSP(metric=False, n_jobs=3, dissimilarity="precomputed")
        mds_clf.fit(sim)

        mds = MDSP()
        init = np.array([[2], [3], [1], [1]])
        mds.fit_transform(sim, init=init)
        mds.fit_transform(sim)

    def test_smacof_p_error(self):
        # Not symmetric similarity matrix:
        sim = np.array(
            [[0, 5, 9, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]]
        )

        np.testing.assert_raises(ValueError, smacof_p, sim, n_uq=1)

        # Not squared similarity matrix:
        sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [4, 2, 1, 0]])

        np.testing.assert_raises(ValueError, smacof_p, sim, n_uq=1)

        # init not None and not correct format:
        sim = np.array(
            [[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]]
        )

        Z = np.array([[-0.266, -0.539], [0.016, -0.238], [-0.200, 0.524]])
        np.testing.assert_raises(
            ValueError, smacof_p, sim, n_uq=1, init=Z, n_init=1
        )

    def test_smacof_p(self):
        sim = np.array(
            [[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]]
        )
        smacof_p(sim, n_uq=1, return_n_iter=False)
