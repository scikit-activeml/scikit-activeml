import unittest

import numpy as np
from sklearn.exceptions import DataConversionWarning

from skactiveml.classifier import (
    ParzenWindowClassifier,
    SlidingWindowClassifier,
)
from skactiveml.pool import UncertaintySampling
from skactiveml.regressor import NICKernelRegressor


class TestColumnVectorTargetContract(unittest.TestCase):
    def test_classifier_accepts_resolved_single_output_column_vector(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([[0], [1], [-1], [-1]])

        target_cases = [("auto", [0, 1]), ("single-output", None)]
        for target_type, classes in target_cases:
            with self.subTest(target_type=target_type, classes=classes):
                classifier = ParzenWindowClassifier(
                    classes=classes,
                    missing_label=-1,
                    random_state=0,
                    target_type=target_type,
                )

                with self.assertWarns(DataConversionWarning):
                    classifier.fit(X, y)

                self.assertEqual(
                    classifier.target_spec_.target_type, "single-output"
                )
                np.testing.assert_array_equal(classifier.classes_, [0, 1])

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ParzenWindowClassifier(missing_label=-1).fit(X, y)

    def test_classifier_backed_query_accepts_single_output_column_vector(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([[0], [1], [-1], [-1]])
        classifier = ParzenWindowClassifier(
            classes=[0, 1], missing_label=-1, random_state=0
        )
        strategy = UncertaintySampling(missing_label=-1, random_state=0)

        with self.assertWarns(DataConversionWarning):
            query_indices = strategy.query(X, y, classifier, batch_size=1)

        self.assertEqual(query_indices.shape, (1,))

    def test_column_vector_is_single_output_without_multilabel_support(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([[0], [1], [-1], [-1]])
        classifier = SlidingWindowClassifier(
            estimator=ParzenWindowClassifier(missing_label=-1),
            missing_label=-1,
            random_state=0,
        )

        with self.assertWarns(DataConversionWarning):
            query_indices = UncertaintySampling(missing_label=-1).query(
                X, y, classifier
            )
        self.assertEqual(query_indices.shape, (1,))

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            UncertaintySampling(missing_label=-1).query(
                X, y, ParzenWindowClassifier(missing_label=-1)
            )

    def test_regressor_accepts_single_output_column_vector(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([[0.5], [1.5], [np.nan], [np.nan]])

        regressor = NICKernelRegressor(random_state=0).fit(X, y)

        self.assertEqual(regressor.target_spec_.target_type, "single-output")
        self.assertEqual(regressor.predict(X[:1]).shape, (1,))
