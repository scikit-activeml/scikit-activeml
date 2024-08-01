import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
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
        super().setUp(
            qs_class=Badge,
            init_default_params={"random_state": 42},
            query_default_params_clf=self.query_default_params_clf,
        )

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

        self.assertEqual(
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
