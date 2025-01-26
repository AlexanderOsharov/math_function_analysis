import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Latex

class MathFunctionAnalysis:
    def __init__(self, function_str):
        self.x = sp.symbols('x')
        self.function = sp.sympify(function_str)
        self.domain = self._find_domain()
        self.intercepts = self._find_intercepts()
        self.asymptotes = self._find_asymptotes()
        self.first_derivative_analysis = self._analyze_first_derivative()
        self.second_derivative_analysis = self._analyze_second_derivative()

    def _find_domain(self):
        try:
            domain = sp.calculus.util.continuous_domain(self.function, self.x, sp.S.Reals)
        except Exception as e:
            domain = f"Error determining domain: {e}"
        return domain

    def _find_intercepts(self):
        x_intercepts = sp.solve(self.function, self.x)
        y_intercept = self.function.subs(self.x, 0)
        return {'x': x_intercepts, 'y': y_intercept}

    def _find_asymptotes(self):
        vertical_asymptotes = sp.calculus.util.vertical_asymptotes(self.function, self.x)
        horizontal_asymptotes = (sp.limit(self.function, self.x, sp.oo), sp.limit(self.function, self.x, -sp.oo))
        oblique_asymptotes = sp.apl.uplimits.oblique(self.function, self.x)
        return {
            'vertical': vertical_asymptotes,
            'horizontal': horizontal_asymptotes,
            'oblique': oblique_asymptotes
        }

    def _analyze_first_derivative(self):
        first_derivative = sp.diff(self.function, self.x)
        critical_points = sp.solve(first_derivative, self.x)
        increasing_intervals = sp.calculus.util.function_monotone_intervals(first_derivative, self.x, relational=False)
        decreasing_intervals = sp.calculus.util.function_monotone_intervals(-first_derivative, self.x, relational=False)
        extrema_values = {point: self.function.subs(self.x, point) for point in critical_points}
        return {
            'derivative': first_derivative,
            'critical_points': critical_points,
            'increasing_intervals': increasing_intervals,
            'decreasing_intervals': decreasing_intervals,
            'extrema_values': extrema_values
        }

    def _analyze_second_derivative(self):
        second_derivative = sp.diff(self.function, self.x, 2)
        inflection_points = sp.solve(second_derivative, self.x)
        concave_up_intervals = sp.calculus.util.function_monotone_intervals(second_derivative, self.x, relational=False)
        concave_down_intervals = sp.calculus.util.function_monotone_intervals(-second_derivative, self.x, relational=False)
        inflection_values = {point: self.function.subs(self.x, point) for point in inflection_points}
        return {
            'derivative': second_derivative,
            'inflection_points': inflection_points,
            'concave_up_intervals': concave_up_intervals,
            'concave_down_intervals': concave_down_intervals,
            'inflection_values': inflection_values
        }

    def plot(self):
        f = sp.lambdify(self.x, self.function, 'numpy')
        x_vals = sp.calculus.util.continuous_domain(self.function, self.x, sp.Interval(-10, 10))
        x_vals = [val.evalf() for val in x_vals]
        y_vals = [f(val) for val in x_vals]

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='Function')

        if self.intercepts['y']:
            plt.scatter([0], [self.intercepts['y']], color='red', label='Y-intercept')
        for intercept in self.intercepts['x']:
            plt.scatter([intercept], [0], color='green', label='X-intercept' if self.intercepts['y'] == 0 else '')

        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.title('Function Analysis Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def report(self):
        report = (
            f"Domain: {self.domain}\n"
            f"Intercepts: {self.intercepts}\n"
            f"Asymptotes: {self.asymptotes}\n"
            f"First Derivative Analysis: {self.first_derivative_analysis}\n"
            f"Second Derivative Analysis: {self.second_derivative_analysis}\n"
        )
        return report

    def display_report(self):
        display(Latex(f"**Domain:** {sp.latex(self.domain)}"))
        display(Latex(f"**Intercepts:** X: {', '.join(map(str, self.intercepts['x']))}, Y: {self.intercepts['y']}"))
        display(Latex(f"**Asymptotes:** Vertical: {', '.join(map(str, self.asymptotes['vertical']))}, "
                      f"Horizontal: {self.asymptotes['horizontal']}, Oblique: {self.asymptotes['oblique']}"))
        display(Latex(f"**First Derivative Analysis:** Derivative: {sp.latex(self.first_derivative_analysis['derivative'])}, "
                      f"Critical Points: {', '.join(map(str, self.first_derivative_analysis['critical_points']))}, "
                      f"Increasing Intervals: {self.first_derivative_analysis['increasing_intervals']}, "
                      f"Decreasing Intervals: {self.first_derivative_analysis['decreasing_intervals']}, "
                      f"Extrema Values: {self.first_derivative_analysis['extrema_values']}"))
        display(Latex(f"**Second Derivative Analysis:** Derivative: {sp.latex(self.second_derivative_analysis['derivative'])}, "
                      f"Inflection Points: {', '.join(map(str, self.second_derivative_analysis['inflection_points']))}, "
                      f"Concave Up Intervals: {self.second_derivative_analysis['concave_up_intervals']}, "
                      f"Concave Down Intervals: {self.second_derivative_analysis['concave_down_intervals']}, "
                      f"Inflection Values: {self.second_derivative_analysis['inflection_values']}"))