try:
    import unittest
    import random
    from copy import deepcopy
    from itertools import combinations

    import numpy as np
    from scipy.special import softmax
    from sklearn.datasets import make_blobs
    from sklearn.utils.validation import check_is_fitted, NotFittedError

    import torch
    from torch import nn

    from skorch.utils import to_numpy

    from skactiveml.tests.template_estimator import TemplateEstimator
    from skactiveml.classifier.multiannotator import CrowdLayerClassifier

    class TestCrowdLayerClassifier(TemplateEstimator, unittest.TestCase):

        def setUp(self):
            # Set global seeds.
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            # Synthetic multi-class data.
            self.X, self.y_true = make_blobs(
                n_samples=300, n_features=2, centers=3, random_state=1
            )
            self.classes = np.unique(self.y_true)
            self.n_classes = len(self.classes)
            self.n_annotators = 10
            self.missing_label = np.nan

            # Build multi-annotator labels (10 annotators).
            rng = np.random.RandomState(1)
            self.y_annot = np.empty((self.X.shape[0], self.n_annotators))
            for m in range(self.n_annotators):
                self.y_annot[:, m] = self.y_true

            # annotator 0: always wrong, annotator 1: random
            self.y_annot[:, 0] = (self.y_true + 1) % self.n_classes
            self.y_annot[:, 1] = rng.choice(
                self.classes, size=len(self.y_true)
            )

            # sprinkle missing labels
            mask = rng.rand(*self.y_annot.shape) < 0.1
            self.y_annot[mask] = self.missing_label
            self.y_annot_unlbld = np.full_like(
                self.y_annot, self.missing_label
            )

            # Skorch config kept tiny so tests run fast.
            neural_net_param_dict = {
                "train_split": None,
                "verbose": 0,
                "optimizer": torch.optim.RAdam,
                "device": "cpu",
                "max_epochs": 5,
                "batch_size": 8,
                "lr": 0.01,
            }

            init_default_params = {
                "clf_module": TestNeuralNet,
                "n_annotators": self.n_annotators,
                "neural_net_param_dict": neural_net_param_dict,
                "sample_dtype": np.float32,
                "classes": self.classes,
                "cost_matrix": None,
                "missing_label": self.missing_label,
                "random_state": 0,
            }

            # Define default parameters for fitting.
            fit_default_params = {
                "X": self.X,
                "y": self.y_annot,
            }

            # Define default parameters for predicting.
            predict_default_params = {"X": self.X}

            super().setUp(
                estimator_class=CrowdLayerClassifier,
                init_default_params=init_default_params,
                fit_default_params=fit_default_params,
                predict_default_params=predict_default_params,
            )

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _make_init_params(self, **overrides):
            params = deepcopy(self.init_default_params)
            params.update(overrides)
            return params

        def _train_for_output_tests(
            self, max_epochs=50, module__return_embeddings=True
        ):
            init_params = self._make_init_params()
            init_params["neural_net_param_dict"]["max_epochs"] = max_epochs
            init_params["neural_net_param_dict"][
                "module__return_embeddings"
            ] = module__return_embeddings
            clf = CrowdLayerClassifier(**init_params)
            clf.fit(self.X, self.y_annot)
            return clf

        def _check_predict_outputs(self, out, mode, n_features):
            # Normalize to a dict so we can selectively check keys
            n_samples = self.X.shape[0]

            # primary output: probabilities or labels
            if mode == "proba":
                # Try common key names, fall back to "first" if
                # using tuple layout.
                P_class = out["proba"]
                self.assertIsNotNone(
                    P_class, "No probability output found for 'proba' mode."
                )
                self.assertEqual(P_class.shape, (n_samples, self.n_classes))
                np.testing.assert_array_almost_equal(
                    P_class.sum(axis=-1), np.ones((n_samples,))
                )

                # If logits are present, check that softmax(logits) == P_class.
                if "logits" in out:
                    L_class = out["logits"]
                    self.assertEqual(
                        L_class.shape, (n_samples, self.n_classes)
                    )
                    np.testing.assert_array_almost_equal(
                        softmax(L_class, axis=-1), P_class
                    )

            elif mode == "label":
                y_class = out["label"]
                self.assertIsNotNone(
                    y_class, "No label output found for 'label' mode."
                )
                self.assertEqual(y_class.shape, (n_samples,))
                self.assertTrue(np.isin(y_class, self.classes).all())

                if "logits" in out:
                    L_class = out["logits"]
                    self.assertEqual(
                        L_class.shape, (n_samples, self.n_classes)
                    )
                    np.testing.assert_array_almost_equal(
                        np.argmax(L_class, axis=-1), y_class
                    )
            else:
                self.fail(f"Unknown mode {mode!r} in _check_predict_outputs")

            # Embeddings of samples
            if "embeddings" in out:
                X_embed = out["embeddings"]
                self.assertEqual(X_embed.shape, (n_samples, n_features))

            # Annotator performance
            if "annotator_perf" in out:
                P_perf = out["annotator_perf"]
                self.assertEqual(P_perf.shape, (n_samples, self.n_annotators))
                self.assertGreaterEqual(P_perf.min(), 0)
                self.assertLessEqual(P_perf.max(), 1)

                # Good annotators (2..9) should outperform always-wrong
                # + random (0,1).
                mean_perf = P_perf.mean(axis=0)
                good_mean = mean_perf[2:].mean()
                bad_mean = mean_perf[:2].mean()
                self.assertGreater(
                    good_mean,
                    bad_mean,
                    msg=(
                        "Good annotators not clearly better:"
                        f" good={good_mean:.3f}, bad={bad_mean:.3f}"
                    ),
                )

        def _test_extra_outputs(self, predict_method):
            test_cases = [
                ("proba", ValueError),
                (["proba"], ValueError),
                (False, TypeError),
                (None, None),
                ([], None),
            ]
            items = [
                "logits",
                "embeddings",
                "annotator_perf",
                "annotator_class",
            ]

            all_combinations = [
                list(combo)
                for r in range(1, len(items) + 1)
                for combo in combinations(items, r)
            ]
            for comb in all_combinations:
                test_cases.append((comb, None))
            self._test_param(
                predict_method,
                "extra_outputs",
                test_cases,
                extras_params={"X": self.X},
            )

        # ------------------------------------------------------------------
        # __init__ parameter tests
        # ------------------------------------------------------------------

        def test_init_param_clf_module(self):
            test_cases = [
                (TestNeuralNet, None),
                (TestNeuralNet(), None),
                ("Test", TypeError),
            ]
            self._test_param("init", "clf_module", test_cases)

        def test_init_param_n_annotators(self):
            # None: allowed, can be inferred from y
            test_cases = [
                (None, None),
                (10, None),
                ("test", ValueError),
                (0, ValueError),
            ]
            self._test_param("init", "n_annotators", test_cases)

        def test_init_param_neural_net_param_dict(self):
            # Must be dict-like or None; conflicting keys should error
            good_dict = {
                "max_epochs": 1,
                "batch_size": 4,
            }
            bad_type = "not_a_dict"
            bad_conflict = {
                "module__clf_module": "not_a_module",
            }
            bad_value = {"train_split": True}

            test_cases = [
                (None, None),
                (good_dict, None),
                (bad_type, TypeError),
                (bad_conflict, TypeError),
                (bad_value, ValueError),
                ({"predict_nonlinearity": nn.Identity()}, ValueError),
                ({"module": TestNeuralNet}, ValueError),
                ({"criterion": nn.CrossEntropyLoss}, ValueError),
            ]
            self._test_param("init", "neural_net_param_dict", test_cases)

        def test_init_param_sample_dtype(self):
            test_cases = [
                (None, RuntimeError),
                (np.float32, None),
                (np.int32, RuntimeError),
            ]
            self._test_param("init", "sample_dtype", test_cases)

        def test_init_param_cost_matrix(self):
            test_cases = [
                (1 - np.eye(self.n_classes), None),
                (1 - np.eye(self.n_classes + 1), ValueError),
                ("test", ValueError),
                (None, None),
            ]
            self._test_param("init", "cost_matrix", test_cases)

        def test_init_param_classes(self):
            test_cases = [
                (None, None),
                (np.arange(self.n_classes), None),
                (np.arange(1, self.n_classes + 1), ValueError),
                ("abc", TypeError),
            ]
            self._test_param("init", "classes", test_cases)

        # ------------------------------------------------------------------
        # predict / predict_proba parameter tests
        # ------------------------------------------------------------------

        def test_predict_param_extra_outputs(self):
            self._test_extra_outputs("predict")

        def test_predict_proba_param_extra_outputs(self):
            self._test_extra_outputs("predict_proba")

        # ------------------------------------------------------------------
        # Output logic tests for predict_proba / predict
        # ------------------------------------------------------------------

        def test_predict_proba(self):
            clf = self._train_for_output_tests()
            extra_outputs_list = [
                [
                    "annotator_class",
                ],
                [
                    "logits",
                    "embeddings",
                    "annotator_perf",
                    "annotator_class",
                ],
            ]
            for extra_outputs in extra_outputs_list:
                out = clf.predict_proba(
                    self.X,
                    extra_outputs=extra_outputs,
                )
                out = dict(zip(["proba"] + extra_outputs, out))
                self._check_predict_outputs(out, mode="proba", n_features=128)

        def test_predict(self):
            clf = self._train_for_output_tests()
            extra_outputs_list = [
                [
                    "annotator_class",
                    "annotator_perf",
                ],
                [
                    "annotator_perf",
                    "annotator_class",
                    "embeddings",
                    "logits",
                ],
            ]
            for extra_outputs in extra_outputs_list:
                out = clf.predict(
                    self.X,
                    extra_outputs=extra_outputs,
                )
                out = dict(zip(["label"] + extra_outputs, out))
                self._check_predict_outputs(out, mode="label", n_features=128)

        # ------------------------------------------------------------------
        # partial_fit / initialize / fit behavior
        # ------------------------------------------------------------------

        def test_partial_fit(self):
            # Case 1: classes=None and only unlabeled data → error
            init_params = self._make_init_params(classes=None)
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            self.assertRaises(
                ValueError, clf.partial_fit, self.X, self.y_annot_unlbld
            )

            # Case 2: unlabeled first, then labeled
            init_params = self._make_init_params()
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf.partial_fit(self.X, self.y_annot_unlbld)
            clf.partial_fit(self.X, self.y_annot)
            check_is_fitted(clf)

            predict_proba_0 = clf.predict_proba(self.X)
            clf.partial_fit(self.X, self.y_annot_unlbld)
            predict_proba_1 = clf.predict_proba(self.X)
            np.testing.assert_almost_equal(predict_proba_0, predict_proba_1)

        def test_initialize(self):
            # Prediction w/o initialization but with default params.
            init_params = self._make_init_params()
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            y_pred = clf.predict(self.X)
            self.assertTrue(np.isin(y_pred, self.classes).all())

            # Prediction with explicit initialization.
            init_params = self._make_init_params()
            clf = CrowdLayerClassifier(**init_params)
            clf.initialize()
            y_pred = clf.predict(self.X)
            self.assertTrue(np.isin(y_pred, self.classes).all())

            # Check that initialization fails without set classes.
            init_params = self._make_init_params()
            init_params["classes"] = None
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(ValueError, clf.initialize)

            # Check that initialization fails without set n_annotators.
            init_params = self._make_init_params()
            init_params["n_annotators"] = None
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(ValueError, clf.initialize)

        def test_fit(self):
            # Check standard fitting cases.
            init_params = self._make_init_params(classes=None)
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            self.assertRaises(ValueError, clf.fit, self.X, self.y_annot_unlbld)

            init_params = self._make_init_params()
            clf = CrowdLayerClassifier(**init_params)
            clf.fit(self.X, self.y_annot)
            check_is_fitted(clf)

            # Check fitting without warm_start (weights should change).
            init_params = self._make_init_params()
            init_params["neural_net_param_dict"]["warm_start"] = False
            clf = CrowdLayerClassifier(**init_params)
            clf.fit(self.X, self.y_annot_unlbld)
            init_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            clf.fit(self.X, self.y_annot_unlbld)
            new_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Check fitting with warm_start (weights unchanged on unlabeled).
            init_params = self._make_init_params()
            init_params["neural_net_param_dict"]["warm_start"] = True
            clf = CrowdLayerClassifier(**init_params)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf.fit(self.X, self.y_annot_unlbld)
            check_is_fitted(clf)
            init_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            clf.fit(self.X, self.y_annot_unlbld)
            new_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            np.testing.assert_array_equal(init_weights, new_weights)
            clf.fit(self.X, self.y_annot)
            new_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Setup for initialized PyTorch module as input.
            init_params = self._make_init_params()
            clf_module = TestNeuralNet()
            init_weights = to_numpy(
                deepcopy(clf_module.input_to_hidden.weight)
            )
            init_params["clf_module"] = clf_module
            clf = CrowdLayerClassifier(**init_params)

            # Fitting with only unlabeled data must preserve weights.
            clf.fit(self.X, self.y_annot_unlbld)
            new_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            np.testing.assert_array_equal(init_weights, new_weights)

            # Fitting with partially labeled data must change weights.
            clf.fit(self.X, self.y_annot)
            new_weights = to_numpy(
                deepcopy(
                    clf.neural_net_.module_.clf_module.input_to_hidden.weight
                )
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Check minimum accuracy requirement.
            init_params = self._make_init_params()
            init_params["neural_net_param_dict"]["max_epochs"] = 50
            clf = CrowdLayerClassifier(**init_params)
            clf.fit(self.X, self.y_annot)
            acc = clf.score(X=self.X, y=self.y_true)
            self.assertGreaterEqual(
                acc,
                0.8,
                msg=f"Accuracy {acc:.3f} must be >= 0.8",
            )

    class TestNeuralNet(nn.Module):
        """Simple 2D → 3-class MLP used as clf_module in tests."""

        def __init__(self, dropout_rate=0.1, return_embeddings=True):
            super().__init__()
            self.input_to_hidden = nn.Linear(2, 128, bias=True)
            self.hidden_to_output = nn.Linear(128, 3, bias=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.return_embeddings = return_embeddings

        def forward(self, X):
            hidden = torch.relu(self.input_to_hidden(X))
            logits = self.hidden_to_output(self.dropout(hidden))
            if self.return_embeddings:
                return logits, hidden
            else:
                return logits

except ImportError:
    # torch/skorch not available -> silently skip tests
    pass  # pragma: no cover
