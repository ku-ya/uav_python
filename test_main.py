#!/usr/bin/env python
import unittest
from main import *

class testMain(unittest.TestCase):
    """Tests for `main.py`."""
    def setUp(self):
        pass
    def test_vee(self):
        """Testing vee function"""
        mat = np.reshape(range(9),(3,3))
        np.testing.assert_array_equal(vee(mat),[7,2,3])
    def test_hat(self):
        """Testing hat function"""
        mat = np.matrix('0 -3 2;3 0 -1;-2 1 0')
        np.testing.assert_array_equal(hat([1,2,3]),mat)

if __name__ == '__main__':
    unittest.main()
