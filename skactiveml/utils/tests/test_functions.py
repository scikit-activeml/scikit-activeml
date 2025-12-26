import unittest

import inspect

from skactiveml.utils import call_func, match_signature

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.utils import make_criterion_tuple_aware

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover


class TestFunctions(unittest.TestCase):
    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)

        # test kwargs
        def sum_all(*args, **kwargs):
            s = 0
            for v in args + tuple(kwargs.values()):
                s += v
            return s

        def test_func_0(**kwargs):
            return sum_all(**kwargs)

        def test_func_1(arg1, **kwargs):
            return sum_all(arg1, **kwargs)

        def test_func_2(kwarg1=0, **kwargs):
            return sum_all(kwarg1=kwarg1, **kwargs)

        result = call_func(test_func_0, arg1=1, arg2=2, arg3=3)
        self.assertEqual(result, 6)

        result = call_func(test_func_1, arg1=1, arg2=2, arg3=3)
        self.assertEqual(result, 6)

        result = call_func(test_func_2, kwarg1=1, arg2=2, arg3=3)
        self.assertEqual(result, 6)

        result = call_func(
            test_func_1, only_mandatory=True, arg1=1, arg2=2, arg3=3
        )
        self.assertEqual(result, 1)

        result = call_func(
            test_func_2, only_mandatory=True, kwarg1=1, arg2=2, arg3=3
        )
        self.assertEqual(result, 0)

        result = call_func(
            test_func_2, ignore_var_keyword=True, kwarg1=1, arg2=2, arg3=3
        )
        self.assertEqual(result, 1)

    def test_match_signature(self):
        class DummyA:
            def __init__(self, dummy_b):
                self.dummy_b = dummy_b

            # test case where c has to be in kwargs to not fail
            @match_signature("dummy_b", "test_me")
            def test_me(self, a, b=None, **kwargs):
                return self.dummy_b.test_me(a=a, b=b, **kwargs)

            # test case without kwargs
            @match_signature("dummy_b", "test_me_alt")
            def test_me_alt(self, a, b=None, **kwargs):
                return self.dummy_b.test_me_alt(a=a, **kwargs)

            # test case without kwargs
            @match_signature("dummy_b", "test_me_hidden")
            def test_me_hidden(self, a, b=None, **kwargs):
                return self.dummy_b.test_me_alt(a=a, **kwargs)

            # test case with type hinting
            @match_signature("dummy_b", "test_me_hint")
            def test_me_hint(self, a, b=None, **kwargs):
                return self.dummy_b.test_me_alt(a=a, **kwargs)

        class DummyB:
            def test_me(self, a, c, **kwargs):
                output = {"a": a, "c": c}
                output.update(kwargs)
                return output

            def test_me_alt(self, a, c):
                output = {"a": a, "c": c}
                return output

            def test_me_hint(self, a: int, c: str) -> dict:
                output = {"a": a, "c": c}
                return output

        dummy_b = DummyB()
        dummy_a = DummyA(dummy_b)

        # test default working case
        kwargs_1 = {
            "a": "p1",
            "b": "p2",
            "c": "p3",
            "d": "p4",
        }
        output_1 = dummy_a.test_me(**kwargs_1)
        self.assertEqual(kwargs_1, output_1)

        # test for equal signature
        sig_a_test_me = inspect.signature(dummy_a.test_me).parameters
        sig_b_test_me = inspect.signature(dummy_b.test_me).parameters
        self.assertEqual(sig_a_test_me, sig_b_test_me)

        # test non working case with missing c
        kwargs_2 = {
            "a": "p1",
            "b": "p2",
            "d": "p4",
        }
        self.assertRaises(TypeError, dummy_a.test_me, **kwargs_2)

        # test for equal signature
        sig_a_test_me_alt = inspect.signature(dummy_a.test_me_alt).parameters
        sig_b_test_me_alt = inspect.signature(dummy_b.test_me_alt).parameters
        self.assertEqual(sig_a_test_me_alt, sig_b_test_me_alt)

        kwargs_3 = {
            "a": "p1",
            "c": "p2",
        }
        dummy_a.test_me_alt(**kwargs_3)

        kwargs_3 = {
            "a": "p1",
            "b": "p2",
            "c": "p3",
        }
        self.assertRaises(TypeError, dummy_a.test_me_alt, **kwargs_3)

        # test for hiding methods that the wrapped object does not have
        self.assertFalse(hasattr(dummy_a, "test_me_hidden"))

        sig_a_test_me_hint = inspect.signature(dummy_a.test_me_hint)
        sig_b_test_me_hint = inspect.signature(dummy_b.test_me_hint)

        self.assertEqual(
            sig_a_test_me_hint.return_annotation,
            sig_b_test_me_hint.return_annotation,
        )

        param_iterator = zip(
            sig_a_test_me_hint.parameters.items(),
            sig_b_test_me_hint.parameters.items(),
        )
        for (name_a, param_a), (name_b, param_b) in param_iterator:
            self.assertEqual(name_a, name_b)
            self.assertEqual(param_a.kind, param_b.kind)
            self.assertEqual(param_a.annotation, param_a.annotation)
            self.assertEqual(param_a.default, param_a.default)

    if successful_skorch_torch_import:

        def test_make_criterion_tuple_aware(self):
            x = torch.randn(8, 5)
            target = torch.randint(0, 5, (8,))

            base_loss = nn.CrossEntropyLoss()
            loss = base_loss(x, target)

            # Invalid `criterion` (hits TypeError in the first block).
            class NotModule:
                pass

            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(NotModule)

            # No selection at all: both None -> return original (class).
            self.assertIs(
                make_criterion_tuple_aware(nn.CrossEntropyLoss),
                nn.CrossEntropyLoss,
            )

            # No selection at all: both None -> return original (instance).
            crit_instance = nn.CrossEntropyLoss()
            self.assertIs(
                make_criterion_tuple_aware(crit_instance),
                crit_instance,
            )

            # Invalid combination: criterion_output_keys not None
            # & forward_outputs None.
            with self.assertRaises(ValueError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs=None,
                )

            # Invalid forward_outputs type (non-dict).
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs=[("logits", (0, None))],
                )

            # criterion_output_keys=None & forward_outputs={} -> StopIteration
            # -> ValueError.
            with self.assertRaises(ValueError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys=None,
                    forward_outputs={},
                )

            # Common mapping for multiple tests.
            forward_outputs = {
                "logits": (0, None),
                "embeddings": (1, None),
            }

            # Name-based: criterion_output_keys is str.
            TupleAwareCE = make_criterion_tuple_aware(
                nn.CrossEntropyLoss,
                criterion_output_keys="logits",
                forward_outputs=forward_outputs,
            )

            # Class should be a subclass of base and cached by name.
            self.assertTrue(issubclass(TupleAwareCE, nn.CrossEntropyLoss))

            # Signature equality (class constructor).
            sig1 = inspect.signature(TupleAwareCE).parameters
            sig2 = inspect.signature(nn.CrossEntropyLoss).parameters
            self.assertEqual(sig1, sig2)

            # Signature equality (forward).
            sig1 = inspect.signature(TupleAwareCE.forward).parameters
            sig2 = inspect.signature(nn.CrossEntropyLoss.forward).parameters
            self.assertEqual(sig1, sig2)

            # Same behavior for tensor input: wrapper just forwards unchanged.
            out = TupleAwareCE()(x, target)
            self.assertTrue(torch.allclose(loss, out))

            # For tuple input, only index 0 ("logits") is used.
            out_tuple = TupleAwareCE()((x, x + 1), target)
            self.assertTrue(torch.allclose(loss, out_tuple))

            # Instance path: wrap an initialized criterion.
            ce_instance = nn.CrossEntropyLoss(ignore_index=-10)
            wrapped_instance = make_criterion_tuple_aware(
                ce_instance,
                criterion_output_keys="logits",
                forward_outputs=forward_outputs,
            )
            # We get a new instance with a different class but still behaves
            # like CE.
            self.assertIsInstance(wrapped_instance, nn.CrossEntropyLoss)
            self.assertEqual(wrapped_instance.ignore_index, -10)
            self.assertIsNot(wrapped_instance, ce_instance)
            out_inst = wrapped_instance((x, x + 1), target)
            self.assertTrue(torch.allclose(loss, out_inst))

            # Caching: same selector -> same subclass object.
            TupleAwareCE2 = make_criterion_tuple_aware(
                nn.CrossEntropyLoss,
                criterion_output_keys="logits",
                forward_outputs=forward_outputs,
            )
            self.assertIs(TupleAwareCE, TupleAwareCE2)

            # criterion_output_keys is a non-empty sequence of strings.
            class DummyTupleLoss(nn.Module):
                def forward(self, input, target=None):
                    # store to inspect what we received
                    self.last_input = input
                    if isinstance(input, tuple):
                        return input[0].sum() + 10 * input[1].sum()
                    return input.sum()

            forward_outputs_dummy = {
                "a": (0, None),
                "b": (1, None),
                "c": (2, None),
            }

            TupleAwareDummy = make_criterion_tuple_aware(
                DummyTupleLoss,
                criterion_output_keys=("a", "c"),
                forward_outputs=forward_outputs_dummy,
            )

            a = torch.ones(3, 3)
            b = torch.full((3, 3), 2.0)
            c = torch.full((3, 3), 3.0)

            dummy_loss = TupleAwareDummy()
            out_dummy = dummy_loss((a, b, c), target=None)
            expected = a.sum() + 10 * c.sum()
            self.assertTrue(torch.allclose(out_dummy, expected))
            # Ensure selector=(0,2) branch is used and passes a tuple of
            # (a, c).
            self.assertIsInstance(dummy_loss.last_input, tuple)
            self.assertEqual(len(dummy_loss.last_input), 2)
            self.assertTrue(torch.allclose(dummy_loss.last_input[0], a))
            self.assertTrue(torch.allclose(dummy_loss.last_input[1], c))

            # Empty sequence for criterion_output_keys -> ValueError.
            with self.assertRaises(ValueError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys=[],
                    forward_outputs={"logits": (0, None)},
                )

            # Non-string entries in sequence -> TypeError.
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys=["logits", 1],
                    forward_outputs={"logits": (0, None)},
                )

            # criterion_output_keys of wrong type (e.g. int) -> TypeError.
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys=123,
                    forward_outputs={"logits": (0, None)},
                )

            # Unknown name in criterion_output_keys -> ValueError.
            with self.assertRaises(ValueError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="unknown",
                    forward_outputs={"logits": (0, None)},
                )

            # Bad spec: not a tuple -> TypeError.
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs={"logits": 123},
                )

            # Bad spec: wrong length -> TypeError.
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs={"logits": (0,)},
                )

            # Bad spec: first element not int -> TypeError.
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs={"logits": ("0", None)},
                )

            # .Bad spec: negative idx -> ValueError.
            with self.assertRaises(ValueError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs={"logits": (-1, None)},
                )

            # Bad spec: transform not callable / None -> TypeError.
            with self.assertRaises(TypeError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys="logits",
                    forward_outputs={"logits": (0, "not_callable")},
                )

            # Duplicate indices across different names -> ValueError.
            forward_outputs_dup = {
                "logits": (0, None),
                "proba": (0, None),
            }
            with self.assertRaises(ValueError):
                make_criterion_tuple_aware(
                    nn.CrossEntropyLoss,
                    criterion_output_keys=("logits", "proba"),
                    forward_outputs=forward_outputs_dup,
                )

            # Sanity check: selector attr exists and matches indices for the
            # last valid class.
            self.assertTrue(
                hasattr(TupleAwareDummy, "_criterion_output_selector")
            )
            self.assertEqual(
                TupleAwareDummy._criterion_output_selector, (0, 2)
            )
            self.assertEqual(
                TupleAwareDummy.__name__, "TupleAwareDummyTupleLossIdx0_2"
            )
