import unittest
from math_function_analysis.analysis import MathFunctionAnalysis

class TestMathFunctionAnalysis(unittest.TestCase):
    def setUp(self):
        self.analysis = MathFunctionAnalysis("1 / (x**2 - 1)")

    def test_domain(self):
        self.assertEqual(self.analysis.domain, sp.Union(sp.Interval.open(-sp.oo, -1), sp.Interval.open(-1, 1), sp.Interval.open(1, sp.oo)))

    def test_intercepts(self):
        self.assertEqual(self.analysis.intercepts['x'], [-1, 1])
        self.assertEqual(self.analysis.intercepts['y'], 0)

    def test_asymptotes(self):
        self.assertEqual(self.analysis.asymptotes['vertical'], [-1, 1])
        self.assertEqual(self.analysis.asymptotes['horizontal'], (0, 0))
        self.assertEqual(self.analysis.asymptotes['oblique'], None)

    def test_first_derivative_analysis(self):
        self.assertEqual(self.analysis.first_derivative_analysis['derivative'], -2/(x**3 - x**2) + 2/(x**3 + x**2))
        self.assertEqual(self.analysis.first_derivative_analysis['critical_points'], [])
        self.assertEqual(self.analysis.first_derivative_analysis['increasing_intervals'], sp.Union(sp.Interval.open(-sp.oo, -1), sp.Interval.open(1, sp.oo)))
        self.assertEqual(self.analysis.first_derivative_analysis['decreasing_intervals'], sp.EmptySet)
        self.assertEqual(self.analysis.first_derivative_analysis['extrema_values'], {})

    def test_second_derivative_analysis(self):
        self.assertEqual(self.analysis.second_derivative_analysis['derivative'], 6/(x**4 - 2*x**2 + 1) - 6/(x**4 + 2*x**2 + 1))
        self.assertEqual(self.analysis.second_derivative_analysis['inflection_points'], [-sqrt(2)/2, sqrt(2)/2])
        self.assertEqual(self.analysis.second_derivative_analysis['concave_up_intervals'], sp.Union(sp.Interval.open(-sqrt(2)/2, 0), sp.Interval.open(sqrt(2)/2, sp.oo)))
        self.assertEqual(self.analysis.second_derivative_analysis['concave_down_intervals'], sp.Union(sp.Interval.open(-sp.oo, -sqrt(2)/2), sp.Interval.open(0, sqrt(2)/2)))
        self.assertEqual(self.analysis.second_derivative_analysis['inflection_values'], {-sqrt(2)/2: -sqrt(2), sqrt(2)/2: sqrt(2)})

if __name__ == '__main__':
    unittest.main()