try:
    import unittest
    import numpy as np
    from types import SimpleNamespace

    import torch
    from torch import nn

    from skactiveml.classifier.multiannotator._utils import (
        _SkorchMultiAnnotatorClassifier,
        _MultiAnnotatorClassificationModule,
        _MultiAnnotatorCollate,
    )

    class _LogitsOnlyNet(nn.Module):
        """Backbone that returns only logits."""

        def __init__(self, in_dim=5, out_dim=3):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.linear(x)

    class _LogitsAndEmbedNet(nn.Module):
        """Backbone that returns (logits, embedding)."""

        def __init__(self, in_dim=5, embed_dim=7, out_dim=3):
            super().__init__()
            self.embed = nn.Linear(in_dim, embed_dim)
            self.classifier = nn.Linear(embed_dim, out_dim)

        def forward(self, x):
            h = torch.relu(self.embed(x))
            logits = self.classifier(h)
            return logits, h

    class _DummyMultiAnnotatorModule(nn.Module):
        """
        Minimal multi-annotator module used for
        `_SkorchMultiAnnotatorClassifier` tests.
        """

        def __init__(
            self,
            n_classes=None,
            n_annotators=None,
            clf_module=None,
            clf_module_param_dict=None,
        ):
            super().__init__()
            # We do not actually use these.
            # They just test that parameters are wired correctly.
            self.n_classes = n_classes
            self.n_annotators = n_annotators
            self.clf_module = clf_module
            self.clf_module_param_dict = clf_module_param_dict

        def forward(self, x, *args, **kwargs):
            # Not used in tests.
            return x

    class _DummyClfModule(nn.Module):
        """Dummy backbone used as `clf_module`."""

        def __init__(self, in_dim=2, out_dim=3):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.linear(x)

    class _DummySkorchMultiAnnotator(_SkorchMultiAnnotatorClassifier):
        """
        Concrete subclass for testing `_SkorchMultiAnnotatorClassifier`
        internals.
        """

        def _build_neural_net_param_overrides(self, X, y):
            # Pretend the module needs n_classes and n_annotators
            return {
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
            }

    class TestSkorchMultiAnnotatorClassifierInternals(unittest.TestCase):
        def setUp(self):
            rng = np.random.RandomState(0)
            self.n_samples = 10
            self.n_annotators = 3
            self.n_features = 2
            self.X = rng.randn(self.n_samples, self.n_features).astype(
                np.float32
            )
            self.y = rng.randint(
                0, 2, size=(self.n_samples, self.n_annotators)
            )
            self.classes = np.array([0, 1])

            self.base_init_params = dict(
                multi_annotator_module=_DummyMultiAnnotatorModule,
                criterion=nn.CrossEntropyLoss,
                clf_module=_DummyClfModule,
                n_annotators=self.n_annotators,
                neural_net_param_dict=None,
                sample_dtype=np.float32,
                classes=self.classes,
                cost_matrix=None,
                missing_label=-1,
                random_state=0,
            )

        # ------------------------------------------------------------------
        # _net_parts: happy path
        # ------------------------------------------------------------------

        def test_net_parts_happy_path(self):
            # `neural_net_param_dict=None` -> should create dict with
            # `¨train_split¨=None`
            clf = _DummySkorchMultiAnnotator(**self.base_init_params)
            module, criterion, nn_params = clf._net_parts(self.X, self.y)

            # module and criterion are just passed through
            self.assertIs(clf.module, module)
            self.assertIs(clf.criterion, criterion)

            # train_split must be present and None
            self.assertIn("train_split", nn_params)
            self.assertIsNone(nn_params["train_split"])

            # invariant keys must be present
            self.assertIs(nn_params["module__clf_module"], clf.clf_module)
            clf_module_param_dict = nn_params["module__clf_module_param_dict"]
            self.assertIsInstance(clf_module_param_dict, dict)

            # overrides from subclass must be merged
            self.assertEqual(nn_params["module__n_classes"], len(self.classes))
            self.assertEqual(
                nn_params["module__n_annotators"], self.n_annotators
            )

            # attributes must be set
            self.assertTrue(hasattr(clf, "classes_"))
            self.assertTrue(hasattr(clf, "n_annotators_"))
            np.testing.assert_array_equal(clf.classes_, self.classes)
            self.assertEqual(clf.n_annotators_, self.n_annotators)

            # dict without "train_split" triggers the defaulting branch
            neural_net_param_dict = {
                "max_epochs": 3,
                "batch_size": 4,
            }
            params = dict(
                self.base_init_params,
                neural_net_param_dict=neural_net_param_dict,
            )
            clf = _DummySkorchMultiAnnotator(**params)

            _, _, nn_params = clf._net_parts(self.X, self.y)

            # "train_split" must have been injected and set to `None`
            self.assertIn("train_split", nn_params)
            self.assertIsNone(nn_params["train_split"])
            # other keys preserved
            self.assertEqual(nn_params["max_epochs"], 3)
            self.assertEqual(nn_params["batch_size"], 4)

        # ------------------------------------------------------------------
        # `_net_parts`: `neural_net_param_dict` handling
        # ------------------------------------------------------------------

        def test_net_parts_conflicting_user_param_raises(self):
            # Subclass that adds an override on a NON-module__ key
            class ConflictOverride(_DummySkorchMultiAnnotator):
                def _build_neural_net_param_overrides(self, X, y):
                    # Start from the base overrides to keep behavior realistic
                    base = super()._build_neural_net_param_overrides(X, y)
                    # Add a non-module__ key that can conflict with user params
                    base["max_epochs"] = 1
                    return base

            # User provides a different value for the same key
            neural_net_param_dict_conflict = {
                "train_split": None,
                "max_epochs": 999,
                # will conflict with `override["max_epochs"] == 1`
            }
            params_conflict = dict(
                self.base_init_params,
                neural_net_param_dict=neural_net_param_dict_conflict,
            )
            clf_conflict = ConflictOverride(**params_conflict)

            with self.assertRaises(ValueError):
                clf_conflict._net_parts(self.X, self.y)

        def test_net_parts_neural_net_param_dict_invalid_type(self):
            params = dict(
                self.base_init_params, neural_net_param_dict="not_a_dict"
            )
            clf = _DummySkorchMultiAnnotator(**params)
            with self.assertRaises(TypeError):
                clf._net_parts(self.X, self.y)

        def test_net_parts_train_split_must_be_none(self):
            neural_net_param_dict = {"train_split": lambda x: x}
            params = dict(
                self.base_init_params,
                neural_net_param_dict=neural_net_param_dict,
            )
            clf = _DummySkorchMultiAnnotator(**params)
            with self.assertRaises(ValueError):
                clf._net_parts(self.X, self.y)

        # ------------------------------------------------------------------
        # `_net_parts`: `classes` / `n_annotators` validation
        # ------------------------------------------------------------------

        def test_net_parts_missing_classes_raises(self):
            params = dict(self.base_init_params, classes=None)
            clf = _DummySkorchMultiAnnotator(**params)
            with self.assertRaises(ValueError):
                clf._net_parts(self.X, self.y)

        def test_net_parts_missing_n_annotators_and_y_raises(self):
            params = dict(self.base_init_params, n_annotators=None)
            clf = _DummySkorchMultiAnnotator(**params)
            with self.assertRaises(ValueError):
                clf._net_parts(self.X, y=None)

        def test_net_parts_n_annotators_mismatch_raises(self):
            params = dict(
                self.base_init_params, n_annotators=self.n_annotators + 1
            )
            clf = _DummySkorchMultiAnnotator(**params)
            with self.assertRaises(ValueError):
                clf._net_parts(self.X, self.y)

        def test_net_parts_override_must_return_dict(self):
            class BadOverride(_DummySkorchMultiAnnotator):
                def _build_neural_net_param_overrides(self, X, y):
                    return "not_a_dict"

            clf = BadOverride(**self.base_init_params)
            with self.assertRaises(TypeError):
                clf._net_parts(self.X, self.y)

        def test_net_parts_override_illegal_keys(self):
            class IllegalOverride(_DummySkorchMultiAnnotator):
                def _build_neural_net_param_overrides(self, X, y):
                    # Try to override invariant key
                    return {"module__clf_module": object()}

            clf = IllegalOverride(**self.base_init_params)
            with self.assertRaises(ValueError):
                clf._net_parts(self.X, self.y)

        # ------------------------------------------------------------------
        # `_return_training_data`
        # ------------------------------------------------------------------

        def test_return_training_data_some_labeled(self):
            clf = _DummySkorchMultiAnnotator(**self.base_init_params)

            # Patch a dummy neural_net_ with a `module_` that has
            # set_`forward_return`
            class DummyForwardModule:
                def __init__(self):
                    self.called = False

                def set_forward_return(self, values=None):
                    self.called = True

            dummy_module = DummyForwardModule()
            clf.neural_net_ = SimpleNamespace(module_=dummy_module)

            # `y` with some -1 (unlabeled) rows
            y = np.array(
                [
                    [0, -1, -1],
                    [1, 1, -1],
                    [-1, -1, -1],
                    [0, 1, 1],
                ],
                dtype=float,
            )
            X = self.X[:4]

            X_train, y_train = clf._return_training_data(X, y)

            # rows 0,1,3 have at least one labeled entry
            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(y_train.shape, (3, self.n_annotators))
            self.assertTrue(dummy_module.called)
            self.assertTrue(np.issubdtype(y_train.dtype, np.integer))

        def test_return_training_data_all_unlabeled(self):
            clf = _DummySkorchMultiAnnotator(**self.base_init_params)

            class DummyForwardModule:
                def __init__(self):
                    self.called = False

                def set_forward_return(self, values=None):
                    self.called = True

            dummy_module = DummyForwardModule()
            clf.neural_net_ = SimpleNamespace(module_=dummy_module)

            y = np.full((4, self.n_annotators), -1, dtype=float)
            X = self.X[:4]

            X_train, y_train = clf._return_training_data(X, y)
            self.assertIsNone(X_train)
            self.assertIsNone(y_train)
            # `set_forward_return` should not be called if there is no
            # training data
            self.assertFalse(dummy_module.called)

        # ------------------------------------------------------------------
        # `_validate_data_kwargs`
        # ------------------------------------------------------------------

        def test_validate_data_kwargs_sets_y_ensure_1d_false(self):
            clf = _DummySkorchMultiAnnotator(**self.base_init_params)
            vd_kwargs = clf._validate_data_kwargs()
            self.assertIsInstance(vd_kwargs, dict)
            self.assertIn("y_ensure_1d", vd_kwargs)
            self.assertFalse(vd_kwargs["y_ensure_1d"])

    class TestMultiAnnotatorClassificationModule(unittest.TestCase):
        def setUp(self):
            self.batch_size = 4
            self.in_dim = 5
            self.n_classes = 3
            self.x = torch.randn(self.batch_size, self.in_dim)

            self.full_outputs = ["logits_class", "x_embed", "logits_annot"]

        # ------------------------------------------------------------------
        # `_as_module` tests
        # ------------------------------------------------------------------

        def test_as_module_with_instance(self):
            backbone = _LogitsOnlyNet(
                in_dim=self.in_dim, out_dim=self.n_classes
            )
            mod = _MultiAnnotatorClassificationModule(
                clf_module=backbone,
                clf_module_param_dict={"in_dim": 999},  # should be ignored
                default_forward_outputs=["logits_class"],
                full_forward_outputs=self.full_outputs,
            )
            # Instance should be used as-is, not reconstructed
            self.assertIs(mod.clf_module, backbone)

        def test_as_module_with_class_and_kwargs(self):
            mod = _MultiAnnotatorClassificationModule(
                clf_module=_LogitsOnlyNet,
                clf_module_param_dict={
                    "in_dim": self.in_dim,
                    "out_dim": self.n_classes,
                },
                default_forward_outputs=["logits_class"],
                full_forward_outputs=self.full_outputs,
            )
            self.assertIsInstance(mod.clf_module, _LogitsOnlyNet)

        def test_as_module_invalid_type_raises(self):
            with self.assertRaises(TypeError):
                _MultiAnnotatorClassificationModule._as_module(
                    "not_a_module", {}
                )

        # ------------------------------------------------------------------
        # `set_forward_return` tests
        # ------------------------------------------------------------------

        def test_set_forward_return_default_and_string(self):
            # `default_forward_outputs` as string
            mod = _MultiAnnotatorClassificationModule(
                clf_module=_LogitsOnlyNet,
                clf_module_param_dict={
                    "in_dim": self.in_dim,
                    "out_dim": self.n_classes,
                },
                default_forward_outputs="logits_class",
                full_forward_outputs=self.full_outputs,
            )
            # Called in `__init__` with None -> should pick default
            self.assertEqual(mod.forward_return, {"logits_class"})

            # single string value
            ret = mod.set_forward_return("x_embed")
            self.assertIs(ret, mod)
            self.assertEqual(mod.forward_return, {"x_embed"})

        def test_set_forward_return_list_and_subset(self):
            mod = _MultiAnnotatorClassificationModule(
                clf_module=_LogitsOnlyNet,
                clf_module_param_dict={
                    "in_dim": self.in_dim,
                    "out_dim": self.n_classes,
                },
                default_forward_outputs=["logits_class"],
                full_forward_outputs=self.full_outputs,
            )

            # multiple valid names
            mod.set_forward_return(["logits_class", "x_embed"])
            self.assertEqual(mod.forward_return, {"logits_class", "x_embed"})

        def test_set_forward_return_unknown_raises(self):
            mod = _MultiAnnotatorClassificationModule(
                clf_module=_LogitsOnlyNet,
                clf_module_param_dict={
                    "in_dim": self.in_dim,
                    "out_dim": self.n_classes,
                },
                default_forward_outputs=["logits_class"],
                full_forward_outputs=self.full_outputs,
            )
            with self.assertRaises(ValueError):
                mod.set_forward_return(["logits_class", "unknown"])

        # ------------------------------------------------------------------
        # `clf_module_forward` tests
        # ------------------------------------------------------------------

        def test_clf_module_forward_logits_only(self):
            mod = _MultiAnnotatorClassificationModule(
                clf_module=_LogitsOnlyNet,
                clf_module_param_dict={
                    "in_dim": self.in_dim,
                    "out_dim": self.n_classes,
                },
                default_forward_outputs=["logits_class"],
                full_forward_outputs=self.full_outputs,
            )

            logits, x_embed = mod.clf_module_forward(self.x)

            # logits should be `(batch_size, n_classes)`
            self.assertEqual(logits.shape, (self.batch_size, self.n_classes))
            # fallback path: `x_embed` should be the original input tensor
            self.assertTrue(torch.allclose(x_embed, self.x))

        def test_clf_module_forward_logits_and_embed(self):
            embed_dim = 7
            mod = _MultiAnnotatorClassificationModule(
                clf_module=_LogitsAndEmbedNet,
                clf_module_param_dict={
                    "in_dim": self.in_dim,
                    "embed_dim": embed_dim,
                    "out_dim": self.n_classes,
                },
                default_forward_outputs=["logits_class"],
                full_forward_outputs=self.full_outputs,
            )

            logits, x_embed = mod.clf_module_forward(self.x)

            self.assertEqual(logits.shape, (self.batch_size, self.n_classes))
            self.assertEqual(x_embed.shape, (self.batch_size, embed_dim))
            # make sure it is not just the raw input passed through
            self.assertNotEqual(x_embed.shape, self.x.shape)

    class TestMultiAnnotatorCollate(unittest.TestCase):

        def setUp(self):
            self.n_samples = 3
            self.n_annotators = 2
            self.feature_dim = 5

            # Simple `X`: just to check that it is collated correctly
            self.X = [
                torch.arange(self.feature_dim, dtype=torch.float32) + i
                for i in range(self.n_samples)
            ]

        # --------------------------------------------------------------
        # Helper to build `(x, y)` batch for a given `y` matrix
        # --------------------------------------------------------------

        def _make_batch(self, y_matrix):
            if isinstance(y_matrix, np.ndarray):
                y_matrix = torch.from_numpy(y_matrix)
            # one row per sample
            return [(self.X[i], y_matrix[i]) for i in range(self.n_samples)]

        # --------------------------------------------------------------
        # Tests
        # --------------------------------------------------------------

        def test_int_missing_label(self):
            # `y`:
            # [ [0,  1],
            #   [1, -1],
            #   [-1, -1] ]
            # valid pairs: (s=0,a=0) -> 0, (0,1) -> 1, (1,0) -> 1
            y = np.array(
                [
                    [0, 1],
                    [1, -1],
                    [-1, -1],
                ],
                dtype=np.int64,
            )
            batch = self._make_batch(y)

            collate = _MultiAnnotatorCollate(missing_label=-1)
            x_out, y_flat = collate(batch)

            # x_out["x"] must be stacked X
            self.assertEqual(
                x_out["x"].shape, (self.n_samples, self.feature_dim)
            )
            # verify stacking order
            for i in range(self.n_samples):
                self.assertTrue(torch.allclose(x_out["x"][i], self.X[i]))

            # Check indices and labels
            input_ids = x_out["input_ids"]
            self.assertEqual(input_ids.shape, (3, 2))
            self.assertEqual(y_flat.shape, (3,))

            # Expected order given the construction in collate:
            # idx_s = [0,0,1,1,2,2]
            # idx_a = [0,1,0,1,0,1]
            # y_flat (unmasked) = [0,1,1,-1,-1,-1]
            # mask keeps first 3 entries.
            expected_idx_s = torch.tensor([0, 0, 1], dtype=torch.long)
            expected_idx_a = torch.tensor([0, 1, 0], dtype=torch.long)
            expected_y = torch.tensor([0, 1, 1], dtype=y_flat.dtype)

            self.assertTrue(torch.equal(input_ids[:, 0], expected_idx_s))
            self.assertTrue(torch.equal(input_ids[:, 1], expected_idx_a))
            self.assertTrue(torch.equal(y_flat, expected_y))

        def test_nan_missing_label(self):
            # Same logical pattern as above, but with NaN labels and a NaN
            # missing_label to trigger the NaN branch.
            y = np.array(
                [
                    [0.0, 1.0],
                    [1.0, np.nan],
                    [np.nan, np.nan],
                ],
                dtype=np.float32,
            )
            batch = self._make_batch(y)

            collate = _MultiAnnotatorCollate(missing_label=float("nan"))
            x_out, y_flat = collate(batch)

            input_ids = x_out["input_ids"]
            self.assertEqual(input_ids.shape, (3, 2))
            self.assertEqual(y_flat.shape, (3,))

            expected_idx_s = torch.tensor([0, 0, 1], dtype=torch.long)
            expected_idx_a = torch.tensor([0, 1, 0], dtype=torch.long)
            expected_y = torch.tensor([0.0, 1.0, 1.0], dtype=y_flat.dtype)

            self.assertTrue(torch.equal(input_ids[:, 0], expected_idx_s))
            self.assertTrue(torch.equal(input_ids[:, 1], expected_idx_a))
            self.assertTrue(torch.equal(y_flat, expected_y))

        def test_all_missing(self):
            # All labels missing -> no pairs.
            y = np.full(
                (self.n_samples, self.n_annotators), -1, dtype=np.int64
            )
            batch = self._make_batch(y)

            collate = _MultiAnnotatorCollate(missing_label=-1)
            x_out, y_flat = collate(batch)

            # x is still the stacked X
            self.assertEqual(
                x_out["x"].shape, (self.n_samples, self.feature_dim)
            )
            input_ids = x_out["input_ids"]
            self.assertEqual(input_ids.shape, (0, 2))
            self.assertEqual(y_flat.shape, (0,))

except ImportError:  # pragma: no cover
    pass
