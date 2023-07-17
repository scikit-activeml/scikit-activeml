import unittest

from sklearn.utils import metaestimators

from skactiveml.utils import call_func


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
