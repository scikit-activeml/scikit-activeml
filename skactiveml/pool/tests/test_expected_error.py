import unittest

import numpy as np

from skactiveml.classifier import PWC
from skactiveml.pool import ExpectedErrorReduction as EER
from skactiveml.utils import MISSING_LABEL


class TestExpectedErrorReduction(unittest.TestCase):

    def setUp(self):
        self.X_cand = np.zeros((100, 2))
        self.X = np.zeros((6, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.cost_matrix = np.eye(3)
        self.clf = PWC(classes=self.classes)

    # Test init parameters
    def test_init_param_method(self):
        eer = EER(method='wrong_method')
        self.assertTrue(hasattr(eer, 'method'))
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, clf=self.clf,
            X=self.X, y=self.y
        )

        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(method=method)
            self.assertTrue(hasattr(eer, 'method'))
            self.assertEqual(eer.method, method)

    def test_init_param_cost_matrix(self):
        for cost_matrix in [np.ones((2, 3)), 'string', np.ones((2, 2))]:
            eer = EER(cost_matrix=cost_matrix)
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
                clf=self.clf
            )

    def test_init_param_random_state(self):
        eer = EER(random_state='string')
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            clf=self.clf
        )

    def test_init_param_ignore_partial_fit(self):
        eer = EER(ignore_partial_fit=None)
        self.assertRaises(
            TypeError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            clf=self.clf
        )

    # Test query parameters
    def test_query_param_clf(self):
        eer = EER()
        self.assertRaises(
            TypeError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            clf='test'
        )

    def test_query_param_X_cand(self):
        eer = EER(cost_matrix=self.cost_matrix)
        self.assertRaises(
            ValueError, eer.query, X_cand=[], X=[], y=[], clf=self.clf
        )
        self.assertRaises(
            ValueError, eer.query, X_cand=[], X=self.X, y=self.y, clf=self.clf
        )

    def test_query_param_X(self):
        eer = EER(cost_matrix=self.cost_matrix)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand,
                          X=np.ones((5, 3)), y=self.y, clf=self.clf)

    def test_query_param_y(self):
        eer = EER(cost_matrix=self.cost_matrix)
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, X=self.X,
            y=[0, 1, 4, 0, 2, 1], clf=self.clf
        )

    def test_query_param_sample_weight(self):
        eer = EER()
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            sample_weight='string', clf=self.clf
        )
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            sample_weight=np.ones(3), clf=self.clf
        )

    def test_query_param_sample_weight_cand(self):
        eer = EER()
        for sample_weight_cand in ['string', np.ones(3)]:
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
                sample_weight_cand=sample_weight_cand, clf=self.clf
            )

    def test_query_param_batch_size(self):
        eer = EER()
        self.assertRaises(TypeError, eer.query, X_cand=self.X_cand, X=self.X,
                          y=self.y, clf=self.clf, batch_size=1.0)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand, X=self.X,
                          y=self.y, clf=self.clf, batch_size=0)

    def test_query_param_return_utilities(self):
        eer = EER(cost_matrix=self.cost_matrix)
        for return_utilities in [None, [], 0]:
            self.assertRaises(TypeError, eer.query, X_cand=self.X_cand,
                              X=self.X, y=self.y, clf=self.clf,
                              return_utilities=return_utilities)

    def test_query(self):
        # Test methods.
        X = [[0], [1], [2]]
        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(method=method)
            _, utilities = eer.query(
                X_cand=X, X=X, y=[MISSING_LABEL, MISSING_LABEL, MISSING_LABEL],
                clf=PWC(classes=[0, 1]), return_utilities=True
            )
            self.assertEqual(utilities.shape, (1, len(X)))
            self.assertEqual(len(np.unique(utilities)), 1)

            _, utilities = eer.query(
                X_cand=X, X=X, y=[0, 1, MISSING_LABEL], clf=self.clf,
                return_utilities=True
            )
            self.assertGreater(utilities[0, 2], utilities[0, 1])
            self.assertGreater(utilities[0, 2], utilities[0, 0])

        # Test scenario.
        X_cand = [[0], [1], [2], [5]]
        eer = EER()

        _, utilities = eer.query(
            X_cand=X_cand, X=[[1]], y=[0], clf=PWC(classes=[0, 1]),
            return_utilities=True
        )
        np.testing.assert_array_equal(utilities, np.zeros((1, len(X_cand))))
        query_indices = eer.query(
            X_cand=[[0], [100], [200]], X=[[0], [200]], y=[0, 1], clf=self.clf
        )
        np.testing.assert_array_equal(query_indices, [1])

    def test_eer_new(self):
        import numpy as np
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_blobs
        from skactiveml.utils import is_unlabeled, MISSING_LABEL
        from skactiveml.classifier import SklearnClassifier
        from skactiveml.pool import ExpectedErrorReduction, \
            EpistemicUncertainty
        from time import time
        import warnings

        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
            X, y_true = make_blobs(random_state=0, centers=5, n_samples=500,
                                   shuffle=True)
            y_true %= 2
            X = StandardScaler().fit_transform(X)
            y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
            y[:450] = y_true[:450]

            clf = SklearnClassifier(
                MLPClassifier(
                    max_iter=1000, hidden_layer_sizes=[100], random_state=0
                ),
                classes=np.unique(y_true), random_state=0)
            qs = ExpectedErrorReduction(method='csl',
                                        ignore_partial_fit=False,
                                        random_state=0)
            # qs = UncertaintySampling(method='least_confident', random_state=0)
            # qs = FourDS(random_state=0)
            # qs = McPAL(random_state=0)
            # gmm = BayesianGaussianMixture(n_components=5, random_state=0)
            # gmm.fit(X)
            # clf = CMM(mixture_model=gmm, classes=np.unique(y_true))
            qs = EpistemicUncertainty()
            clf = SklearnClassifier(estimator=LogisticRegression(),
                                    classes=np.unique(y_true))

            n_cycles = 5
            for c in range(n_cycles):
                clf.fit(X, y)
                print(f'Score: {clf.score(X, y_true)}')
                unlbld_idx = \
                    np.argwhere(is_unlabeled(y, missing_label=MISSING_LABEL))[
                    :, 0]
                X_cand = X[unlbld_idx]
                t = time()
                query_idx = unlbld_idx[
                    qs.query(X_cand=X_cand, X=X, y=y, clf=clf, batch_size=2)]
                print(f'Time: {time() - t}')
                y[query_idx] = y_true[query_idx]
            clf.fit(X, y)
            print(clf.score(X, y_true))
