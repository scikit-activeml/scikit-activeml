import unittest

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.utils.validation import NotFittedError
from torch import nn
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from skactiveml.classifier.multiannotator import RegCrowdNetClassifier


class TestRegCrowdNetClassifierClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y_true = make_blobs(n_samples=300, random_state=0)
        self.X = self.X.astype(np.float32)
        self.y = np.array([self.y_true, self.y_true], dtype=float).T
        self.y[:100, 0] = -1
        self.default_params = {
            "n_classes": 3,
            "n_annotators": 2,
        }
        self.clf_init_params = {
            "classes": [0, 1, 2],
            "missing_label": -1,
            "cost_matrix": None,
            "random_state": 1,
            "train_split": None,
            "verbose": False,
            "optimizer": torch.optim.RAdam,
            "device": "cpu",
            "max_epochs": 5,
            "batch_size": 1,
            "lr": 0.001,
        }

    def test_init_param_module_gt_embed_x(self):
        clf = RegCrowdNetClassifier(
            module__gt_embed_x="Test",
            **self.default_params,
        )
        self.assertEqual(clf.module__gt_embed_x, "Test")
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        clf = RegCrowdNetClassifier(
            module__gt_embed_x=None,
            **self.default_params,
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        clf = RegCrowdNetClassifier(
            module__gt_net=[("nn.Module", GT_Embed_Net)],
            **self.default_params,
        )
        self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        clf = RegCrowdNetClassifier(
            classes=[0, 1, 2],
            module__gt_net=GT_Embed_Net,
            **self.default_params,
        )
        self.assertRaises(ValueError, clf.fit, X=self.X, y=self.y)

    def test_fit(self):
        gt_embed = GT_Embed_Net()
        gt_output = GT_Output_Net()
        clf = RegCrowdNetClassifier(
            module__gt_embed_x=gt_embed,
            module__gt_output=gt_output,
            **self.default_params,
            **self.clf_init_params,
        )

        np.testing.assert_array_equal([0, 1, 2], clf.classes)
        self.assertRaises(NotFittedError, clf.check_is_fitted)
        clf.fit(self.X, self.y)
        self.assertIsNone(clf.check_is_fitted())

    def test_predict(self):
        gt_embed = GT_Embed_Net()
        gt_output = GT_Output_Net()
        clf = RegCrowdNetClassifier(
            module__gt_embed_x=gt_embed,
            module__gt_output=gt_output,
            **self.default_params,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        y_pred = clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    def test_predict_prob(self):
        gt_embed = GT_Embed_Net()
        gt_output = GT_Output_Net()
        clf = RegCrowdNetClassifier(
            module__gt_embed_x=gt_embed,
            module__gt_output=gt_output,
            **self.default_params,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        proba = clf.predict_proba(self.X[:1])
        self.assertEqual(1, proba.shape[0])
        self.assertEqual(3, proba.shape[1])

    def test_predict_annotator_pref(self):
        gt_embed = GT_Embed_Net()
        gt_output = GT_Output_Net()
        clf = RegCrowdNetClassifier(
            module__gt_embed_x=gt_embed,
            module__gt_output=gt_output,
            **self.default_params,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        annot_pref = clf.predict_annotator_perf()
        self.assertEqual(annot_pref.shape[0], 2)
        self.assertEqual(annot_pref.shape[1], 2)
        confusion_matrix = clf.predict_annotator_perf(return_confusion_matrix=True)
        print(confusion_matrix.shape)
        self.assertEqual(2, confusion_matrix.shape[0])
        self.assertEqual(2, confusion_matrix.shape[1])

    def test_predict_P_annot(self):
        gt_net = TestNeuralNet()
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )
        self.assertRaises(NotFittedError, clf.predict, X=self.X)
        clf.fit(self.X, self.y)
        annot = clf.predict_P_annot(self.X[:2])
        print(annot.shape)
        self.assertEqual(annot.shape[0], 2)
        self.assertEqual(annot.shape[1], 3)
        self.assertEqual(annot.shape[2], 2)

    def test_validation_step(self):
        gt_net = TestNeuralNet()
        valid_ds = Dataset(self.X, self.y)
        self.clf_init_params["train_split"] = predefined_split(valid_ds)
        clf = CrowdLayerClassifier(
            module__gt_net=gt_net,
            **self.clf_init_params,
        )
        clf.fit(self.X, self.y)
        self.assertIsNone(clf.check_is_fitted())


class GT_Embed_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(
            in_features=2, out_features=2, bias=True
        )

    def forward(self, X):
        hidden = self.input_to_hidden(X)
        embed_x = torch.relu(hidden)
        return embed_x


class GT_Output_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_to_output = nn.Linear(
            in_features=2, out_features=3, bias=True
        )

    def forward(self, X):
        output_values = self.hidden_to_output(X)
        return output_values
