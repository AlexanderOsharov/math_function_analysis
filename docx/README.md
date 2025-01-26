# Math Function Analysis Library

This library provides tools for analyzing mathematical functions, including finding the domain, intercepts, asymptotes, and analyzing derivatives to determine intervals of increase/decrease and concavity.

## Installation

To use this library, clone the repository and install it via pip:

```bash
pip install git+https://github.com/AlexanderOsharov/math_function_analysis.git
```

## Using

```python
# Импорт необходимых модулей
from math_function_analysis import MathFunctionAnalysis

# Определение функции
function_str = "1 / (x**2 - 1)"

# Создание объекта анализа
analysis = MathFunctionAnalysis(function_str)

# Получение подробного отчета
analysis.display_report()

# Построение графика функции
analysis.plot()

# Поэтапный анализ
analysis.step_by_step_analysis()
```