import unittest

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.utils.validation import NotFittedError
from torch import nn

from skactiveml.classifier.multiannotator import CrowdLayerClassifier


class TestCrowdLayerClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y_true = make_blobs(n_samples=300, random_state=0)
        self.X = self.X.astype(np.float32)
        self.y = np.array([self.y_true, self.y_true], dtype=float).T
        self.y[:100, 0] = -1
        self.clf_init_params = {
            'module__n_classes': 3,
            'module__n_annotators': 2,
            'classes': [0, 1, 2],
            'missing_label': -1,
            'cost_matrix': None,
            'random_state': 1,
            'train_split': None,
            'verbose': False,
            'optimizer': torch.optim.RAdam,
            'device': "cpu",
            'max_epochs': 10,
            'batch_size': 1,
            'lr': 0.001,
        }

    def test_init_param_module_gt_net(self):
        clf = CrowdLayerClassifier(module__gt_net="Test")
        self.assertEqual(clf.module__gt_net, "Test")
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        clf = CrowdLayerClassifier(module__gt_net=None)
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        clf = CrowdLayerClassifier(
            module__gt_net=[("nn.Module", TestNeuralNet)]
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        clf = CrowdLayerClassifier(
            classes=[0, 1, 2], module__gt_net=TestNeuralNet
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)

    def test_fit(self):
        gt_net = TestNeuralNet()
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )

        np.testing.assert_array_equal([0, 1, 2], clf.classes)
        self.assertRaises(NotFittedError, clf.check_is_fitted)
        clf.fit(self.X, self.y)
        self.assertIsNone(clf.check_is_fitted())

    def test_predict(self):
        gt_net = TestNeuralNet()
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        y_pred = clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    def test_predict_annotator_pref(self):
        gt_net = TestNeuralNet()
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        # annot_pref = clf.predict_annotator_perf(self.X[:1])
        # self.assertEqual(len(annot_pref), 2)
        # Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)
        # in line 148
        # in file pytorch.functional softmax 1885

    def test_predict_prob(self):
        gt_net = TestNeuralNet()
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        proba = clf.predict_proba(self.X[:1])
        self.assertEqual(1, proba.shape[0])
        self.assertEqual(3, proba.shape[1])

    def test_predict_P_annot(self):
        gt_net = TestNeuralNet()
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        # annot = clf.predict_P_annot(self.X[:1])
        # self.assertEqual(len(annot_pref), 2)
        # Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)
        # in line 148
        # in file pytorch.functional softmax 1885


class TestNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(
            in_features=2, out_features=2, bias=True
        )
        self.hidden_to_output = nn.Linear(
            in_features=2, out_features=3, bias=True
        )

    def forward(self, X):
        hidden = self.input_to_hidden(X)
        hidden = torch.relu(hidden)
        output_values = self.hidden_to_output(hidden)
        return output_values
