import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Latex
import numpy as np

class MathFunctionAnalysis:
    def __init__(self, function_str, language='ru'):
        self.x = sp.symbols('x')
        self.function = sp.sympify(function_str)
        self.language = language
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
        vertical_asymptotes = self._find_vertical_asymptotes()
        horizontal_asymptotes = self._find_horizontal_asymptotes()
        oblique_asymptotes = self._find_oblique_asymptotes()
        return {
            'vertical': vertical_asymptotes,
            'horizontal': horizontal_asymptotes,
            'oblique': oblique_asymptotes
        }

    def _find_vertical_asymptotes(self):
        try:
            denom = sp.denom(self.function)
            vertical_asymptotes = sp.solve(denom, self.x)
        except Exception as e:
            vertical_asymptotes = f"Error finding vertical asymptotes: {e}"
        return vertical_asymptotes

    def _find_horizontal_asymptotes(self):
        try:
            limit_pos_inf = sp.limit(self.function, self.x, sp.oo)
            limit_neg_inf = sp.limit(self.function, self.x, -sp.oo)
        except Exception as e:
            limit_pos_inf = f"Error finding horizontal asymptote at +∞: {e}"
            limit_neg_inf = f"Error finding horizontal asymptote at -∞: {e}"
        return (limit_pos_inf, limit_neg_inf)

    def _find_oblique_asymptotes(self):
        try:
            degree_num = sp.degree(sp.numer(self.function))
            degree_den = sp.degree(sp.denom(self.function))
            if degree_num > degree_den:
                quotient, remainder = sp.div(sp.numer(self.function), sp.denom(self.function))
                return quotient
        except Exception as e:
            return f"Error finding oblique asymptote: {e}"
        return None

    def _analyze_first_derivative(self):
        first_derivative = sp.diff(self.function, self.x)
        simplified_first_derivative = sp.simplify(first_derivative)
        critical_points = sp.solve(simplified_first_derivative, self.x)
        increasing_intervals = self._find_monotone_intervals(simplified_first_derivative, self.x)
        decreasing_intervals = self._find_monotone_intervals(-simplified_first_derivative, self.x)
        extrema_values = {point: self.function.subs(self.x, point) for point in critical_points}
        return {
            'derivative': simplified_first_derivative,
            'critical_points': critical_points,
            'increasing_intervals': increasing_intervals,
            'decreasing_intervals': decreasing_intervals,
            'extrema_values': extrema_values
        }

    def _analyze_second_derivative(self):
        second_derivative = sp.diff(self.function, self.x, 2)
        simplified_second_derivative = sp.simplify(second_derivative)
        inflection_points = sp.solve(simplified_second_derivative, self.x)
        concave_up_intervals = self._find_monotone_intervals(simplified_second_derivative, self.x)
        concave_down_intervals = self._find_monotone_intervals(-simplified_second_derivative, self.x)
        inflection_values = {point: self.function.subs(self.x, point) for point in inflection_points}
        return {
            'derivative': simplified_second_derivative,
            'inflection_points': inflection_points,
            'concave_up_intervals': concave_up_intervals,
            'concave_down_intervals': concave_down_intervals,
            'inflection_values': inflection_values
        }

    def _find_monotone_intervals(self, derivative, variable):
        critical_points = sp.solve(derivative, variable)
        critical_points = sorted([cp.evalf() for cp in critical_points if cp.is_real])
        domain_intervals = self._get_domain_intervals(critical_points)
        monotone_intervals = []
        for interval in domain_intervals:
            test_point = (interval.start + interval.end) / 2
            if test_point.is_real:
                test_value = derivative.subs(variable, test_point)
                if test_value.is_real and test_value > 0:
                    monotone_intervals.append(interval)
        return monotone_intervals

    def _get_domain_intervals(self, critical_points):
        domain = self.domain
        intervals = []
        if isinstance(domain, sp.Union):
            for subdomain in domain.args:
                intervals.extend(self._get_intervals(subdomain, critical_points))
        else:
            intervals.extend(self._get_intervals(domain, critical_points))
        return intervals

    def _get_intervals(self, domain, critical_points):
        intervals = []
        start = domain.start
        for cp in critical_points:
            if start < cp < domain.end:
                intervals.append(sp.Interval(start, cp, left_open=True, right_open=True))
                start = cp
        intervals.append(sp.Interval(start, domain.end, left_open=start == domain.start, right_open=domain.end == sp.oo))
        return intervals

    def plot(self):
        f = sp.lambdify(self.x, self.function, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = np.array([f(val) if val not in self.asymptotes['vertical'] else np.nan for val in x_vals])

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=self._translate('Function'))

        if self.intercepts['y']:
            plt.scatter([0], [self.intercepts['y']], color='red', label=self._translate('Y-intercept'))
        for intercept in self.intercepts['x']:
            plt.scatter([intercept], [0], color='green', label=self._translate('X-intercept') if self.intercepts['y'] == 0 else '')

        for va in self.asymptotes['vertical']:
            plt.axvline(va, color='purple', linestyle='--', label=f"{self._translate('Vertical Asymptote at x=')} {va}")

        if self.asymptotes['horizontal'][0] != sp.oo and self.asymptotes['horizontal'][0] != -sp.oo:
            plt.axhline(self.asymptotes['horizontal'][0], color='orange', linestyle='--', label=f"{self._translate('Horizontal Asymptote at y=')} {self.asymptotes['horizontal'][0]}")
        if self.asymptotes['horizontal'][1] != sp.oo and self.asymptotes['horizontal'][1] != -sp.oo:
            plt.axhline(self.asymptotes['horizontal'][1], color='orange', linestyle='--', label=f"{self._translate('Horizontal Asymptote at y=')} {self.asymptotes['horizontal'][1]}")

        if self.asymptotes['oblique']:
            oblique_func = sp.lambdify(self.x, self.asymptotes['oblique'], 'numpy')
            y_oblique = np.array([oblique_func(val) for val in x_vals])
            plt.plot(x_vals, y_oblique, color='brown', linestyle='--', label=self._translate('Oblique Asymptote'))

        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.title(self._translate('Function Analysis Plot'))
        plt.xlabel(self._translate('x'))
        plt.ylabel(self._translate('y'))
        plt.legend()
        plt.show()

    def report(self):
        report = (
            f"{self._translate('Domain')}: {self.domain}\n"
            f"{self._translate('Intercepts')}: {self.intercepts}\n"
            f"{self._translate('Asymptotes')}: {self.asymptotes}\n"
            f"{self._translate('First Derivative Analysis')}: {self.first_derivative_analysis}\n"
            f"{self._translate('Second Derivative Analysis')}: {self.second_derivative_analysis}\n"
        )
        return report

    def display_report(self):
        display(Latex(f"\\textbf{{{self._translate('Domain')}}}: {sp.latex(self.domain)}"))
        display(Latex(f"\\textbf{{{self._translate('Intercepts')}}}: X: {', '.join(map(lambda x: sp.latex(x), self.intercepts['x']))}, Y: {sp.latex(self.intercepts['y'])}"))
        display(Latex(f"\\textbf{{{self._translate('Asymptotes')}}}: {self._translate('Vertical')}: {', '.join(map(lambda x: sp.latex(x), self.asymptotes['vertical']))}, "
                      f"{self._translate('Horizontal')}: {', '.join(map(lambda x: sp.latex(x), self.asymptotes['horizontal']))}, {self._translate('Oblique')}: {sp.latex(self.asymptotes['oblique'])}"))
        display(Latex(f"\\textbf{{{self._translate('First Derivative Analysis')}}}: {self._translate('Derivative')}: {sp.latex(self.first_derivative_analysis['derivative'])}, "
                      f"{self._translate('Critical Points')}: {', '.join(map(lambda x: sp.latex(x), self.first_derivative_analysis['critical_points']))}, "
                      f"{self._translate('Increasing Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.first_derivative_analysis['increasing_intervals']))}, "
                      f"{self._translate('Decreasing Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.first_derivative_analysis['decreasing_intervals']))}, "
                      f"{self._translate('Extrema Values')}: {', '.join(map(lambda x: f'${sp.latex(x)}$: ${sp.latex(self.first_derivative_analysis['extrema_values'][x])}$', self.first_derivative_analysis['extrema_values']))}"))
        display(Latex(f"\\textbf{{{self._translate('Second Derivative Analysis')}}}: {self._translate('Derivative')}: {sp.latex(self.second_derivative_analysis['derivative'])}, "
                      f"{self._translate('Inflection Points')}: {', '.join(map(lambda x: sp.latex(x), self.second_derivative_analysis['inflection_points']))}, "
                      f"{self._translate('Concave Up Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.second_derivative_analysis['concave_up_intervals']))}, "
                      f"{self._translate('Concave Down Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.second_derivative_analysis['concave_down_intervals']))}, "
                      f"{self._translate('Inflection Values')}: {', '.join(map(lambda x: f'${sp.latex(x)}$: ${sp.latex(self.second_derivative_analysis['inflection_values'][x])}$', self.second_derivative_analysis['inflection_values']))}"))

    def step_by_step_analysis(self):
        steps = [
            f"\\textbf{{{self._translate('Domain')}}}: {sp.latex(self.domain)}",
            f"\\textbf{{{self._translate('Intercepts')}}}: X: {', '.join(map(lambda x: sp.latex(x), self.intercepts['x']))}, Y: {sp.latex(self.intercepts['y'])}",
            f"\\textbf{{{self._translate('Asymptotes')}}}: {self._translate('Vertical')}: {', '.join(map(lambda x: sp.latex(x), self.asymptotes['vertical']))}, "
            f"{self._translate('Horizontal')}: {', '.join(map(lambda x: sp.latex(x), self.asymptotes['horizontal']))}, {self._translate('Oblique')}: {sp.latex(self.asymptotes['oblique'])}",
            f"\\textbf{{{self._translate('First Derivative Analysis')}}}:",
            f"\\quad {self._translate('Derivative')}: {sp.latex(self.first_derivative_analysis['derivative'])}",
            f"\\quad {self._translate('Critical Points')}: {', '.join(map(lambda x: sp.latex(x), self.first_derivative_analysis['critical_points']))}",
            f"\\quad {self._translate('Increasing Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.first_derivative_analysis['increasing_intervals']))}",
            f"\\quad {self._translate('Decreasing Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.first_derivative_analysis['decreasing_intervals']))}",
            f"\\quad {self._translate('Extrema Values')}: {', '.join(map(lambda x: f'${sp.latex(x)}$: ${sp.latex(self.first_derivative_analysis["extrema_values"][x])}$', self.first_derivative_analysis['extrema_values']))}",
            f"\\textbf{{{self._translate('Second Derivative Analysis')}}}:",
            f"\\quad {self._translate('Derivative')}: {sp.latex(self.second_derivative_analysis['derivative'])}",
            f"\\quad {self._translate('Inflection Points')}: {', '.join(map(lambda x: sp.latex(x), self.second_derivative_analysis['inflection_points']))}",
            f"\\quad {self._translate('Concave Up Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.second_derivative_analysis['concave_up_intervals']))}",
            f"\\quad {self._translate('Concave Down Intervals')}: {', '.join(map(lambda x: sp.latex(x), self.second_derivative_analysis['concave_down_intervals']))}",
            f"\\quad {self._translate('Inflection Values')}: {', '.join(map(lambda x: f'${sp.latex(x)}$: ${sp.latex(self.second_derivative_analysis["inflection_values"][x])}$', self.second_derivative_analysis['inflection_values']))}"
        ]
        for step in steps:
            display(Latex(step))

    def _translate(self, key):
        translations = {
            'Domain': {'ru': 'Область определения', 'en': 'Domain'},
            'Intercepts': {'ru': 'Пересечения с осями', 'en': 'Intercepts'},
            'Asymptotes': {'ru': 'Асимптоты', 'en': 'Asymptotes'},
            'Vertical': {'ru': 'Вертикальные', 'en': 'Vertical'},
            'Horizontal': {'ru': 'Горизонтальные', 'en': 'Horizontal'},
            'Oblique': {'ru': 'Наклонные', 'en': 'Oblique'},
            'Function': {'ru': 'Функция', 'en': 'Function'},
            'Y-intercept': {'ru': 'Пересечение с осью Y', 'en': 'Y-intercept'},
            'X-intercept': {'ru': 'Пересечение с осью X', 'en': 'X-intercept'},
            'Vertical Asymptote at x=': {'ru': 'Вертикальная асимптота в x=', 'en': 'Vertical Asymptote at x='},
            'Horizontal Asymptote at y=': {'ru': 'Горизонтальная асимптота в y=', 'en': 'Horizontal Asymptote at y='},
            'Oblique Asymptote': {'ru': 'Наклонная асимптота', 'en': 'Oblique Asymptote'},
            'Function Analysis Plot': {'ru': 'График функции', 'en': 'Function Analysis Plot'},
            'x': {'ru': 'x', 'en': 'x'},
            'y': {'ru': 'y', 'en': 'y'},
            'First Derivative Analysis': {'ru': 'Анализ первой производной', 'en': 'First Derivative Analysis'},
            'Derivative': {'ru': 'Производная', 'en': 'Derivative'},
            'Critical Points': {'ru': 'Критические точки', 'en': 'Critical Points'},
            'Increasing Intervals': {'ru': 'Промежутки возрастания', 'en': 'Increasing Intervals'},
            'Decreasing Intervals': {'ru': 'Промежутки убывания', 'en': 'Decreasing Intervals'},
            'Extrema Values': {'ru': 'Значения экстремумов', 'en': 'Extrema Values'},
            'Second Derivative Analysis': {'ru': 'Анализ второй производной', 'en': 'Second Derivative Analysis'},
            'Inflection Points': {'ru': 'Точки перегиба', 'en': 'Inflection Points'},
            'Concave Up Intervals': {'ru': 'Промежутки выпуклости вверх', 'en': 'Concave Up Intervals'},
            'Concave Down Intervals': {'ru': 'Промежутки выпуклости вниз', 'en': 'Concave Down Intervals'},
            'Inflection Values': {'ru': 'Значения в точках перегиба', 'en': 'Inflection Values'}
        }
        return translations.get(key, {}).get(self.language, key)