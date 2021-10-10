import unittest
from unittest.mock import patch

import numpy as np

from skactiveml.base import SkactivemlClassifier, ClassFrequencyEstimator, \
    AnnotModelMixin


class SkactivemlClassifierTest(unittest.TestCase):

    @patch.multiple(SkactivemlClassifier, __abstractmethods__=set())
    def setUp(self):
        self.clf = SkactivemlClassifier(classes=[0, 1], missing_label=-1)

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.clf.fit, X=None, y=None)

    def test_predict_proba(self):
        self.assertRaises(NotImplementedError, self.clf.predict_proba, X=None)

    def test__validate_data(self):
        X = np.ones((10, 2))
        y = np.random.rand(10)
        self.assertRaises(ValueError, self.clf._validate_data, X=X, y=y)


class ClassFrequencyEstimatorTest(unittest.TestCase):

    @patch.multiple(ClassFrequencyEstimator, __abstractmethods__=set())
    def setUp(self):
        self.clf = ClassFrequencyEstimator()

    def test_predict_freq(self):
        self.assertRaises(NotImplementedError, self.clf.predict_freq, X=None)


class AnnotModelMixinTest(unittest.TestCase):

    @patch.multiple(AnnotModelMixin, __abstractmethods__=set())
    def setUp(self):
        self.clf = AnnotModelMixin()

    def test_predict_annot_proba(self):
        self.assertRaises(NotImplementedError, self.clf.predict_annot_proba,
                          X=None)
