from unittest import TestCase

class UtilsTest(TestCase):

    def test_add(self):
        a = 1+1
        self.assertEqual(2, a)
    
    def test_mult(self):
        a = 2 * 3
        self.assertEqual(6, a)
