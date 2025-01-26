import unittest
from math_function_analysis.analysis import MathFunctionAnalysis
import sympy as sp

class TestMathFunctionAnalysis(unittest.TestCase):
    def setUp(self):
        self.analysis = MathFunctionAnalysis("1 / (x**2 - 1)")

    def test_domain(self):
        expected_domain = sp.Union(sp.Interval.open(-sp.oo, -1), sp.Interval.open(-1, 1), sp.Interval.open(1, sp.oo))
        self.assertEqual(self.analysis.domain, expected_domain)

    def test_intercepts(self):
        self.assertEqual(self.analysis.intercepts['x'], [-1, 1])
        self.assertEqual(self.analysis.intercepts['y'], 0)

    def test_asymptotes(self):
        self.assertEqual(self.analysis.asymptotes['vertical'], [-1, 1])
        self.assertEqual(self.analysis.asymptotes['horizontal'], (0, 0))
        self.assertEqual(self.analysis.asymptotes['oblique'], None)

    def test_first_derivative_analysis(self):
        expected_derivative = -2/(x**3 - x**2) + 2/(x**3 + x**2)
        self.assertEqual(self.analysis.first_derivative_analysis['derivative'], expected_derivative)
        self.assertEqual(self.analysis.first_derivative_analysis['critical_points'], [])
        expected_increasing_intervals = sp.Union(sp.Interval.open(-sp.oo, -1), sp.Interval.open(1, sp.oo))
        self.assertEqual(self.analysis.first_derivative_analysis['increasing_intervals'], expected_increasing_intervals)
        self.assertEqual(self.analysis.first_derivative_analysis['decreasing_intervals'], sp.EmptySet)
        self.assertEqual(self.analysis.first_derivative_analysis['extrema_values'], {})

    def test_second_derivative_analysis(self):
        expected_derivative = 6/(x**4 - 2*x**2 + 1) - 6/(x**4 + 2*x**2 + 1)
        self.assertEqual(self.analysis.second_derivative_analysis['derivative'], expected_derivative)
        expected_inflection_points = [-sp.sqrt(2)/2, sp.sqrt(2)/2]
        self.assertEqual(self.analysis.second_derivative_analysis['inflection_points'], expected_inflection_points)
        expected_concave_up_intervals = sp.Union(sp.Interval.open(-sp.sqrt(2)/2, 0), sp.Interval.open(sp.sqrt(2)/2, sp.oo))
        self.assertEqual(self.analysis.second_derivative_analysis['concave_up_intervals'], expected_concave_up_intervals)
        expected_concave_down_intervals = sp.Union(sp.Interval.open(-sp.oo, -sp.sqrt(2)/2), sp.Interval.open(0, sp.sqrt(2)/2))
        self.assertEqual(self.analysis.second_derivative_analysis['concave_down_intervals'], expected_concave_down_intervals)
        expected_inflection_values = {-sp.sqrt(2)/2: -sp.sqrt(2), sp.sqrt(2)/2: sp.sqrt(2)}
        self.assertEqual(self.analysis.second_derivative_analysis['inflection_values'], expected_inflection_values)

if __name__ == '__main__':
    unittest.main()