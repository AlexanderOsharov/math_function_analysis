# Math Function Analysis Library

This library provides tools for analyzing mathematical functions, including finding the domain, intercepts, asymptotes, and analyzing derivatives to determine intervals of increase/decrease and concavity.

## Installation

To use this library, clone the repository and install it via pip:

```bash
pip install git+https://github.com/AlexanderOsharov/math_function_analysis.git
```

## Using

```python
from math_function_analysis import MathFunctionAnalysis

# Define a function
function_str = "1 / (x**2 - 1)"

# Create an analysis object
analysis = MathFunctionAnalysis(function_str)

# Get a detailed report
analysis.display_report()

# Plot the function
analysis.plot()
```