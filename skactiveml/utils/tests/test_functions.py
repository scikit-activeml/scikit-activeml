import unittest

from skactiveml.utils import call_func
from skactiveml.utils._functions import _available_if


class TestFunctions(unittest.TestCase):
    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)

    def test__available_if(self):
        try:
            _available_if("predict_proba", True)
        except ImportError:
            pass
        _available_if("predict_proba", False)
