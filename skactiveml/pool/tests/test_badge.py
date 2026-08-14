import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import Badge
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)

from skactiveml.tests.utils import (
    ParzenWindowClassifierEmbedding,
    ParzenWindowClassifierTuple,
)


class SklearnClassifierCosineEmbedding(SklearnClassifier):
    """`SklearnClassifier` whose `predict_proba` optionally returns a
    `(proba, embeddings)` tuple, where the embeddings are a nonlinear
    transformation of `X` and therefore differ from `X` itself."""

    def predict_proba(self, X, return_embeddings=False):
        probas = super().predict_proba(X)
        if not return_embeddings:
            return probas
        return probas, np.cos(3 * np.asarray(X))


class TestBadge(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        self.classes = [0, 1]
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])

        self.query_default_params_clf = {
            "X": X,
            "y": y,
            "clf": SklearnClassifier(
                LogisticRegression(random_state=0),
                classes=self.classes,
                random_state=0,
            ),
        }
        self.query_default_params_clf_2 = {
            "X": X,
            "y": y,
            "clf": ParzenWindowClassifierEmbedding(
                classes=self.classes, random_state=42
            ),
        }
        self.qs_params_clf_multilabel = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.vstack(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    *[
                        np.full(2, MISSING_LABEL, dtype=float)
                        for _ in range(8)
                    ],
                ]
            ),
            "clf": SklearnClassifier(
                estimator=MultiOutputClassifier(GaussianNB()),
                classes=[[0, 1], [0, 1]],
                missing_label=MISSING_LABEL,
                proba_format="array",
                random_state=0,
            ),
        }
        super().setUp(
            qs_class=Badge,
            init_default_params={"random_state": 42},
            query_default_params_clf=self.query_default_params_clf,
            query_default_params_clf_multilabel=self.qs_params_clf_multilabel,
        )

    def test_target_contract(self):
        self._test_classification_target_contract(
            frozenset(
                {
                    (
                        "classification",
                        "single-output",
                        "single-annotator",
                    ),
                    ("classification", "multi-label", "single-annotator"),
                }
            ),
        )

    def test_fit_clone_keeps_classifier_target_declaration(self):
        X = np.linspace(0, 1, 12).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            target_type="multi-label",
        )

        query_idx = Badge(random_state=0).query(X, y, clf, fit_clf=True)

        self.assertIn(query_idx[0], [4, 5])
        self.assertEqual(clf.target_type, "multi-label")
        self.assertFalse(hasattr(clf, "target_spec_"))

    def test_query_reuses_fitted_target_spec_without_class_evidence(self):
        X = np.linspace(0, 1, 12).reshape(6, 2)
        y_fit = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            target_type="multi-label",
        ).fit(X, y_fit)
        established_spec = clf.target_spec_
        y_query = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                *[[MISSING_LABEL, MISSING_LABEL] for _ in range(4)],
            ]
        )

        query_idx, utilities = Badge(random_state=0).query(
            X, y_query, clf, fit_clf=False, return_utilities=True
        )

        self.assertIn(query_idx[0], [2, 3, 4, 5])
        self.assertIs(clf.target_spec_, established_spec)
        self.assertTrue(np.isnan(utilities[0, :2]).all())

    def _duplicated_pool(self):
        """Builds a degenerate pool whose unlabeled candidates take only two
        distinct feature values."""
        X = np.vstack(
            [
                np.array([[0.0, 0.0], [1.0, 1.0]]),
                np.tile(np.array([[3.0, 3.0], [7.0, 7.0]]), (5, 1)),
            ]
        )
        y = np.hstack([[0, 1], np.full(10, MISSING_LABEL)])
        return X, y

    def test_query_param_clf(self):
        add_test_cases = [
            (SVC(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(SVC(probability=True)), None),
            (
                SklearnClassifier(LogisticRegression(), classes=self.classes),
                None,
            ),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_init_param_clf_embedding_flag_name(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            (None, None),
            (False, TypeError),
            (True, TypeError),
            ("return_embeddings", None),
            ({"return_embeddings": True}, None),
            ({"test": True}, TypeError),
        ]
        self._test_param(
            "init",
            "clf_embedding_flag_name",
            replace_query_params=self.query_default_params_clf_2,
            test_cases=test_cases,
        )

    def test_query(self):
        # test case 1: with the same random state the init pick-up is the same
        badge_1 = Badge(random_state=42)
        X_1 = np.random.RandomState(42).choice(5, size=(10, 2))
        y_1 = np.hstack([[0, 1], np.full(8, MISSING_LABEL)])
        clf_1 = SklearnClassifier(LogisticRegression(), classes=self.classes)

        np.testing.assert_array_equal(
            badge_1.query(X_1, y_1, clf_1), badge_1.query(X_1, y_1, clf_1)
        )

        # test case 2: all utilities are not negative or np.nan
        _, utilities_2 = badge_1.query(
            X_1, y_1, clf_1, batch_size=2, return_utilities=True
        )
        for u in utilities_2:
            for i in u:
                if not np.isnan(i):
                    self.assertGreaterEqual(i, 0)
                else:
                    self.assertTrue(np.isnan(i))

        # test case 3: for the case, the sum of utilities equals to one
        probas = [i for i in utilities_2[0] if not np.isnan(i)]
        probas_sum = np.sum(probas)
        self.assertAlmostEqual(probas_sum, 1)

        probas = [i for i in utilities_2[1] if not np.isnan(i)]
        probas_sum = np.sum(probas)
        self.assertAlmostEqual(probas_sum, 1)

        # test case 4: for candidates.ndim = 1
        candidates_4 = np.arange(4, 10)
        _, utilities_4 = badge_1.query(
            X_1,
            y_1,
            clf_1,
            batch_size=2,
            candidates=candidates_4,
            return_utilities=True,
        )
        for u in utilities_4:
            for i in u:
                if not np.isnan(i):
                    self.assertGreaterEqual(i, 0)
                else:
                    self.assertTrue(np.isnan(i))
        self.assertEqual(2, utilities_4.shape[0])
        self.assertEqual(10, utilities_4.shape[1])

        # test case 5: for candidates with new samples
        X_cand = np.random.choice(5, size=(5, 2))
        _, utilities_5 = badge_1.query(
            X_1,
            y_1,
            clf_1,
            batch_size=2,
            candidates=X_cand,
            return_utilities=True,
        )
        self.assertEqual(5, utilities_5.shape[1])
        self.assertEqual(2, utilities_5.shape[0])

        # test case 6: for clf knows only a single class
        X_6 = np.random.RandomState(42).choice(5, size=(10, 2))
        y_6 = np.hstack([[0], np.full(9, MISSING_LABEL)])
        _, utilities_6 = badge_1.query(
            X_6, y_6, clf_1, batch_size=2, return_utilities=True
        )

        probas = [i for i in utilities_6[0] if not np.isnan(i)]
        probas_sum = np.sum(probas)
        self.assertAlmostEqual(probas_sum, 1)

        probas = [i for i in utilities_6[1] if not np.isnan(i)]
        probas_sum = np.sum(probas)
        self.assertAlmostEqual(probas_sum, 1)

        # test case 7: clf_embedding_flag_name = "return_embeddings"
        clf_7 = ParzenWindowClassifierEmbedding(
            classes=self.classes, random_state=42
        )
        badge_7 = Badge(
            clf_embedding_flag_name="return_embeddings", random_state=42
        )
        np.testing.assert_array_equal(
            badge_7.query(X_1, y_1, clf_7),
            badge_7.query(X_1, y_1, clf_7),
        )

        # test case 8: predict_probas returns tuple
        clf_8 = ParzenWindowClassifierTuple(
            classes=self.classes, random_state=42
        )
        np.testing.assert_array_equal(
            badge_1.query(X_1, y_1, clf_8),
            badge_1.query(X_1, y_1, clf_8),
        )

    def test_query_multilabel(self):
        qs = Badge(random_state=42)
        query_params = dict(self.query_default_params_clf_multilabel)

        query_idx, utilities = qs.query(
            **query_params, batch_size=2, return_utilities=True
        )
        self.assertEqual(len(query_idx), 2)
        self.assertEqual(utilities.shape, (2, len(query_params["X"])))
        self.assertTrue(np.isnan(utilities[:, :2]).all())

        query_idx_2, utilities_2 = qs.query(
            **query_params,
            candidates=np.arange(2, len(query_params["X"])),
            batch_size=2,
            return_utilities=True,
        )
        np.testing.assert_array_equal(query_idx, query_idx_2)
        np.testing.assert_allclose(utilities, utilities_2, equal_nan=True)

    def test_query_multilabel_list_probas(self):
        qs = Badge(random_state=42)
        query_params = dict(self.query_default_params_clf_multilabel)
        query_params["clf"] = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=MISSING_LABEL,
            proba_format="list",
            random_state=0,
        )

        query_idx, utilities = qs.query(
            **query_params, batch_size=2, return_utilities=True
        )
        self.assertEqual(len(query_idx), 2)
        self.assertEqual(utilities.shape, (2, len(query_params["X"])))
        self.assertTrue(np.isnan(utilities[:, :2]).all())

    def test_query_multi_output_multiclass_list_probas_raises(self):
        qs = Badge(random_state=42)
        query_params = {
            "X": np.linspace(0, 1, 12).reshape(6, 2),
            "y": np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [MISSING_LABEL, MISSING_LABEL],
                    [MISSING_LABEL, MISSING_LABEL],
                    [MISSING_LABEL, MISSING_LABEL],
                ]
            ),
            "clf": SklearnClassifier(
                estimator=MultiOutputClassifier(GaussianNB()),
                classes=[[0, 1, 2], [0, 1]],
                missing_label=MISSING_LABEL,
                proba_format="list",
                random_state=0,
            ),
        }

        self.assertRaises(ValueError, qs.query, **query_params)

    def test_query_distinct_indices_per_batch(self):
        # A pool whose candidates take only two distinct feature values must
        # not lead to a sample being selected twice within one batch.
        X, y = self._duplicated_pool()
        clf = SklearnClassifier(LogisticRegression(), classes=self.classes)
        for random_state in range(30):
            query_indices = Badge(random_state=random_state).query(
                X, y, clf, batch_size=6
            )
            self.assertEqual(len(np.unique(query_indices)), 6)

    def test_query_utilities_are_sampling_distributions(self):
        # Each row of `utilities` is the sampling distribution of the
        # respective round, i.e., its `nansum` is one.
        X, y = self._duplicated_pool()
        clf = SklearnClassifier(LogisticRegression(), classes=self.classes)
        _, utilities = Badge(random_state=0).query(
            X, y, clf, batch_size=6, return_utilities=True
        )
        np.testing.assert_allclose(np.nansum(utilities, axis=1), 1)

        X_2, y_2 = make_blobs(
            n_samples=100, centers=2, n_features=4, random_state=1
        )
        y_2 = np.where(np.arange(100) < 6, y_2, MISSING_LABEL).astype(float)
        _, utilities_2 = Badge(random_state=1).query(
            X_2, y_2, clf, batch_size=10, return_utilities=True
        )
        np.testing.assert_allclose(np.nansum(utilities_2, axis=1), 1)

    def test_query_origin_is_no_permanent_center(self):
        # Candidates near the origin of the feature space must not be
        # suppressed once the remaining clusters have been covered.
        cluster_a = np.array(
            [[10.0, 10.0], [10.5, 9.5], [9.5, 10.5], [10.2, 10.2]]
        )
        cluster_b = np.array(
            [[-10.0, -10.0], [-10.5, -9.5], [-9.5, -10.5], [-10.2, -10.2]]
        )
        cluster_c = np.array(
            [[0.2, -0.2], [-0.2, 0.2], [0.1, 0.1], [-0.1, -0.1]]
        )
        X = np.vstack(
            [
                np.array([[3.0, -3.0], [-3.0, 3.0]]),
                cluster_a,
                cluster_b,
                cluster_c,
            ]
        )
        y = np.hstack([[0, 1], np.full(12, MISSING_LABEL)])
        clf = SklearnClassifier(LogisticRegression(), classes=self.classes)
        query_indices, utilities = Badge(random_state=1).query(
            X, y, clf, batch_size=4, return_utilities=True
        )

        # Once the two outer clusters have been sampled from, the uncovered
        # cluster near the origin carries the largest utilities.
        cluster_c_indices = np.arange(10, 14)
        self.assertGreater(
            np.nanmin(utilities[2, cluster_c_indices]),
            np.nanmax(np.delete(utilities[2], cluster_c_indices)),
        )
        self.assertTrue(np.isin(query_indices, cluster_c_indices).any())

    def test_query_matches_reference_kmeanspp(self):
        # The factorized computation agrees per round with the reference
        # semantics evaluated on explicitly materialized gradient embeddings.
        X, y_true = make_blobs(
            n_samples=300, centers=5, n_features=8, random_state=0
        )
        y = np.full(300, MISSING_LABEL)
        lbld_mapping = np.arange(10)
        y[lbld_mapping] = y_true[lbld_mapping] % 2
        batch_size = 8
        clf = SklearnClassifier(
            LogisticRegression(), classes=self.classes, random_state=0
        )
        query_indices, utilities = Badge(random_state=0).query(
            X, y, clf, batch_size=batch_size, return_utilities=True
        )

        # Materialize the `(n_unlabeled, n_classes * n_features)` embeddings.
        unlbld_mapping = np.setdiff1d(np.arange(300), lbld_mapping)
        probas = clf.fit(X, y).predict_proba(X[unlbld_mapping])
        proba_factor = probas - np.eye(probas.shape[1])[probas.argmax(-1)]
        g_x = proba_factor[:, :, None] * X[unlbld_mapping][:, None, :]
        g_x = g_x.reshape(len(unlbld_mapping), -1)
        norms_2 = np.sum(g_x**2, axis=-1)

        # The first center is the argmax of the gradient norms.
        self.assertEqual(query_indices[0], unlbld_mapping[np.argmax(norms_2)])

        query_indices_in_unlbld = [
            int(np.flatnonzero(unlbld_mapping == idx)[0])
            for idx in query_indices
        ]
        d_2 = None
        for i in range(batch_size):
            if i == 0:
                expected = norms_2 / norms_2.sum()
            else:
                idx_in_unlbld = query_indices_in_unlbld[i - 1]
                d_2_new = np.sum((g_x - g_x[idx_in_unlbld]) ** 2, axis=-1)
                d_2 = d_2_new if i == 1 else np.minimum(d_2, d_2_new)
                d_2[query_indices_in_unlbld[:i]] = 0
                expected = d_2 / d_2.sum()

            # Samples selected in an earlier round are `np.nan`, whereas their
            # reference probability is zero.
            row = utilities[i, unlbld_mapping]
            is_selected = np.isnan(row)
            np.testing.assert_array_equal(
                np.flatnonzero(is_selected),
                np.sort(query_indices_in_unlbld[:i]),
            )
            np.testing.assert_allclose(
                row[~is_selected], expected[~is_selected], rtol=1e-9
            )

    def test_query_zero_gradient_embeddings(self):
        # If all gradient embeddings are zero, e.g., for the one-hot
        # probabilities of a single-class cold start, all remaining samples
        # are equally likely and the first one is drawn at random.
        X = np.random.RandomState(42).choice(5, size=(10, 2)).astype(float)
        y = np.hstack([[0], np.full(9, MISSING_LABEL)])
        clf = SklearnClassifier(LogisticRegression(), classes=self.classes)
        first_indices = set()
        for random_state in [0, 1, 42, 123, 2024]:
            query_indices, utilities = Badge(random_state=random_state).query(
                X, y, clf, batch_size=3, return_utilities=True
            )
            first_indices.add(int(query_indices[0]))
            self.assertEqual(len(np.unique(query_indices)), 3)
            np.testing.assert_allclose(utilities[0, 1:], 1 / 9)
        self.assertGreater(len(first_indices), 1)

    def test_query_labeled_candidates(self):
        # A given `candidates` is authoritative, i.e., labeled samples remain
        # candidates, e.g., to relabel them or to recompute their utilities.
        X = np.random.RandomState(0).rand(10, 2)
        y = np.hstack([np.tile([0, 1], 3), np.full(4, MISSING_LABEL)])
        labeled_indices = np.arange(6)
        clf = SklearnClassifier(
            LogisticRegression(), classes=self.classes, random_state=0
        )
        query_indices, utilities = Badge(random_state=0).query(
            X,
            y,
            clf,
            candidates=np.arange(10),
            batch_size=5,
            return_utilities=True,
        )
        self.assertEqual(len(np.unique(query_indices)), 5)
        self.assertFalse(np.isnan(utilities[0]).any())
        np.testing.assert_allclose(np.nansum(utilities, axis=1), 1)

        # Restricting `candidates` to labeled samples selects among those.
        query_indices_lbld = Badge(random_state=0).query(
            X, y, clf, candidates=labeled_indices, batch_size=3
        )
        self.assertTrue(np.isin(query_indices_lbld, labeled_indices).all())
        self.assertEqual(len(np.unique(query_indices_lbld)), 3)

        # In contrast, `candidates=None` considers unlabeled samples only.
        query_indices_none = Badge(random_state=0).query(
            X, y, clf, batch_size=4
        )
        np.testing.assert_array_equal(
            np.sort(query_indices_none), np.arange(6, 10)
        )

        # Without any unlabeled sample, `candidates=None` leaves nothing to
        # select, which is answered with an empty batch.
        with self.assertWarnsRegex(UserWarning, "exhausted"):
            query_indices_exhausted = Badge(random_state=0).query(
                X=X, y=np.tile([0, 1], 5), clf=clf
            )
        self.assertEqual(query_indices_exhausted.shape, (0,))

    def test_query_candidates_as_sample_matrix(self):
        # Candidates that are given as a sample matrix are indexed directly.
        random_state = np.random.RandomState(3)
        X = random_state.rand(20, 4)
        y = np.hstack([[0, 1], np.full(18, MISSING_LABEL)])
        candidates = random_state.rand(15, 4)
        clf = SklearnClassifier(
            LogisticRegression(), classes=self.classes, random_state=3
        )
        query_indices, utilities = Badge(random_state=3).query(
            X,
            y,
            clf,
            candidates=candidates,
            batch_size=5,
            return_utilities=True,
        )
        self.assertEqual(utilities.shape, (5, 15))
        self.assertEqual(len(np.unique(query_indices)), 5)
        self.assertTrue(np.all((query_indices >= 0) & (query_indices < 15)))
        np.testing.assert_allclose(np.nansum(utilities, axis=1), 1)

    def test_query_clf_embedding(self):
        # With `clf_embedding_flag_name`, the sampling distribution is built
        # on the returned embeddings instead of on `X`.
        X = np.random.RandomState(7).rand(30, 3)
        y = np.hstack([[0, 1], np.full(28, MISSING_LABEL)])
        clf = SklearnClassifierCosineEmbedding(
            LogisticRegression(), classes=self.classes, random_state=7
        )
        query_indices, utilities = Badge(
            clf_embedding_flag_name="return_embeddings", random_state=7
        ).query(X, y, clf, batch_size=4, return_utilities=True)
        self.assertEqual(len(np.unique(query_indices)), 4)
        np.testing.assert_allclose(np.nansum(utilities, axis=1), 1)

        # Rebuild the first round from explicitly materialized gradients.
        unlbld_mapping = np.arange(2, 30)
        probas, embeddings = clf.fit(X, y).predict_proba(
            X[unlbld_mapping], return_embeddings=True
        )
        proba_factor = probas - np.eye(probas.shape[1])[probas.argmax(-1)]
        norms_2_emb = np.sum(
            (proba_factor[:, :, None] * embeddings[:, None, :]) ** 2,
            axis=(-2, -1),
        )
        norms_2_X = np.sum(
            (proba_factor[:, :, None] * X[unlbld_mapping][:, None, :]) ** 2,
            axis=(-2, -1),
        )
        self.assertEqual(
            query_indices[0], unlbld_mapping[np.argmax(norms_2_emb)]
        )
        np.testing.assert_allclose(
            utilities[0, unlbld_mapping],
            norms_2_emb / norms_2_emb.sum(),
            rtol=1e-9,
        )
        # The distribution built on `X` differs, such that the embeddings are
        # indeed the ones being used.
        self.assertFalse(
            np.allclose(
                utilities[0, unlbld_mapping], norms_2_X / norms_2_X.sum()
            )
        )
