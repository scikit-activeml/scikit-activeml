import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import NotFittedError

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier.multiannotator import AnnotatorEnsembleClassifier
from skactiveml.utils import MISSING_LABEL
