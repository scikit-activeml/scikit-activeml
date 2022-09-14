import unittest

import numpy as np
from sklearn.svm import SVR

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import CostEmbeddingAL
from skactiveml.pool._cost_embedding_al import MDSP, smacof_p
from skactiveml.utils import MISSING_LABEL


class TestCostEmbeddingAL(unittest.TestCase):
    def setUp(self):
        self.X_cand = np.zeros((100, 2))
        self.X = np.zeros((6, 2))
        self.y = [0, 1, np.nan, np.nan, 2, 1]
        self.classes = [0, 1, 2]
        self.cost_matrix = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
        self.regressor = SVR()
        self.pwc = ParzenWindowClassifier()

    # Test init parameters
    def test_init_param_base_regressor(self):
        alce = CostEmbeddingAL(
            classes=self.classes,
            base_regressor=self.pwc,
            cost_matrix=self.cost_matrix,
        )
        self.assertTrue(hasattr(alce, "base_regressor"))
        self.assertRaises(TypeError, alce.query, self.X, self.y)

    def test_init_param_cost_matrix(self):
        alce = CostEmbeddingAL(classes=self.classes, cost_matrix="A")
        self.assertTrue(hasattr(alce, "cost_matrix"))
        self.assertRaises(ValueError, alce.query, self.X, self.y)

        zero_cost_matrix = np.zeros((len(self.classes), len(self.classes)))
        alce = CostEmbeddingAL(
            classes=self.classes, cost_matrix=zero_cost_matrix
        )
        self.assertRaises(ValueError, alce.query, self.X, self.y)

    def test_init_param_classes(self):
        alce = CostEmbeddingAL(classes=[0, 1], cost_matrix=self.cost_matrix)
        self.assertTrue(hasattr(alce, "classes"))
        self.assertRaises(ValueError, alce.query, self.X, self.y)

    def test_init_param_embed_dim(self):
        alce = CostEmbeddingAL(
            classes=self.classes, cost_matrix=self.cost_matrix, embed_dim=1.5
        )
        self.assertTrue(hasattr(alce, "embed_dim"))
        self.assertRaises(TypeError, alce.query, self.X, self.y)

        alce = CostEmbeddingAL(
            classes=self.classes, cost_matrix=self.cost_matrix, embed_dim=0
        )
        self.assertRaises(ValueError, alce.query, self.X, self.y)

    def test_init_param_missing_label(self):
        alce = CostEmbeddingAL(
            classes=self.classes,
            cost_matrix=self.cost_matrix,
            missing_label=[1, 2, 3],
        )
        self.assertTrue(hasattr(alce, "missing_label"))
        self.assertRaises(TypeError, alce.query, self.X, self.y)

    def test_init_param_mds_params(self):
        alce = CostEmbeddingAL(
            classes=self.classes, cost_matrix=self.cost_matrix, mds_params=0
        )
        self.assertTrue(hasattr(alce, "mds_params"))
        self.assertRaises(TypeError, alce.query, self.X, self.y)

    def test_init_param_nn_params(self):
        alce = CostEmbeddingAL(
            classes=self.classes, cost_matrix=self.cost_matrix, nn_params=0
        )
        self.assertTrue(hasattr(alce, "nn_params"))
        self.assertRaises(TypeError, alce.query, self.X, self.y)

    # Test query parameters
    def test_query_param_X(self):
        alce = CostEmbeddingAL(self.classes, self.regressor, self.cost_matrix)
        self.assertRaises(ValueError, alce.query, X=np.ones((5, 3)), y=self.y)
        _, result = alce.query(
            X=self.X,
            y=[MISSING_LABEL] * len(self.X),
            candidates=self.X_cand,
            return_utilities=True,
        )
        np.testing.assert_array_equal(result, np.ones((1, len(self.X_cand))))

    def test_query_param_y(self):
        alce = CostEmbeddingAL(self.classes, self.regressor, self.cost_matrix)
        self.assertRaises(
            ValueError, alce.query, X=self.X, y=[0, 1, 4, 0, 2, 1]
        )

    def test_query_param_sample_weight(self):
        alce = CostEmbeddingAL(
            classes=self.classes, cost_matrix=self.cost_matrix
        )
        self.assertRaises(
            ValueError, alce.query, X=self.X, y=self.y, sample_weight="string"
        )

    def test_query_param_batch_size(self):
        alce = CostEmbeddingAL(self.classes, self.regressor, self.cost_matrix)
        self.assertRaises(
            TypeError, alce.query, self.X, self.y, batch_size=1.0
        )
        self.assertRaises(ValueError, alce.query, self.X, self.y, batch_size=0)

    def test_query_param_return_utilities(self):
        alce = CostEmbeddingAL(self.classes, self.regressor, self.cost_matrix)
        self.assertRaises(
            TypeError, alce.query, X_cand=self.X_cand, return_utilities=None
        )
        self.assertRaises(
            TypeError, alce.query, X_cand=self.X_cand, return_utilities=[]
        )
        self.assertRaises(
            TypeError, alce.query, X_cand=self.X_cand, return_utilities=0
        )

    def test_query(self):
        alce = CostEmbeddingAL(base_regressor=self.regressor, classes=[0, 1])
        query_indices = alce.query(
            [[0], [200]], [0, 1], candidates=[[0], [100], [200]]
        )
        np.testing.assert_array_equal(query_indices, [1])

    def test_mds_params(self):
        np.random.seed(14)
        X = np.random.random((10, 2))
        y = np.random.randint(0, 2, 10)
        candidates = np.random.random((15, 2))

        alce = CostEmbeddingAL(
            self.classes,
            self.regressor,
            self.cost_matrix,
            random_state=14,
            mds_params={"n_jobs": 1, "verbose": 2},
        )
        cand1 = alce.query(X, y, candidates=candidates)
        alce = CostEmbeddingAL(
            self.classes,
            self.regressor,
            self.cost_matrix,
            random_state=14,
            mds_params={"n_jobs": 2},
        )
        cand2 = alce.query(X, y, candidates=candidates)
        np.testing.assert_array_equal(cand1, cand2)

        alce = CostEmbeddingAL(
            self.classes,
            self.regressor,
            self.cost_matrix,
            mds_params={"dissimilarity": "wrong"},
        )
        self.assertRaises(ValueError, alce.query, X, y, candidates=candidates)

        alce = CostEmbeddingAL(
            base_regressor=self.regressor,
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
