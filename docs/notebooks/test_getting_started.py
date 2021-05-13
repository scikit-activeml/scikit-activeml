import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import is_unlabeled, MISSING_LABEL, plot_2d_dataset
from skactiveml.classifier import SklearnClassifier
from sklearn.metrics import accuracy_score
import warnings
from skactiveml.visualization._feature_space import plot_decision_boundary
warnings.filterwarnings("ignore")


class TestGettingStarted(unittest.TestCase):

    def setUp(self):
        self.random_state = 0
        self.n_features = 2
        self.n_redundant = 0
        self.X = np.make_calssifcation(self.n_features, self.n_redundant, self.random_state)[0]
        self.y_true = np.make_calssifcation(self.n_features, self.n_redundant, self.random_state)[1]
        self.y = np.full(shape=self.y_true.shape, fill_value=MISSING_LABEL)
        self.clf = SklearnClassifier
        self.qs = UncertaintySampling







if __name__ == '__main__':
    unittest.main()
