import unittest
from copy import deepcopy

import numpy as np
from sklearn.linear_model import LogisticRegression

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import Badge
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestBadge(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        self.classes = [0, 1]
