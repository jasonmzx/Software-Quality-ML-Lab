# test.py
import unittest

# calculator.py
class Calculator:
    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y


class TestCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        result = self.calc.add(4, 5)
        self.assertEqual(result, 9)

    def test_subtract(self):
        result = self.calc.subtract(10, 5)
        self.assertEqual(result, 5)

if __name__ == '__main__':
    unittest.main()
