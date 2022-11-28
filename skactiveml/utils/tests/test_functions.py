import unittest

from sklearn.utils import metaestimators

from skactiveml.utils import call_func
from skactiveml.utils._functions import (
    _available_if,
    _IffHasAMethod,
    _hasattr_array_like,
)


class TestFunctions(unittest.TestCase):
    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)

    def test__available_if(self):

        if hasattr(metaestimators, "available_if"):
            wrapper_func = _available_if("a", True)
            self.assertTrue(callable(wrapper_func))

        for method_name in ["a", ("a", "b")]:
            wrapper_func = _available_if(method_name, False)
            self.assertTrue(callable(wrapper_func))

    def test__hasattr_array_like(self):
        class A:
            def __init__(self):
                self.v = "v"

        a = A()

        self.assertTrue(_hasattr_array_like(a, "v"))
        self.assertTrue(_hasattr_array_like(a, ("w", "v")))


class Test_IffHasAMethod(unittest.TestCase):
    def test___get__(self):
        def dummyMethod(x):
            return "method_result"

        class A:
            pass

        class B:
            def do_2(self):
                pass

        class WrapperOfAB:
            wrapped = _IffHasAMethod(
                fn=dummyMethod,
                delegate_name="var",
                method_names=("do_1", "do_2"),
            )

            def __init__(self, var=None):
                self.var = A() if var is None else var

        w = WrapperOfAB()
        self.assertFalse(hasattr(w, "wrapped"))
        b = B()
        b.do_2()
        w = WrapperOfAB(var=b)
        self.assertTrue(hasattr(w, "wrapped"))
        res = w.wrapped()
        self.assertEqual(res, "method_result")
