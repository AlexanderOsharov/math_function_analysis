def is_real_number(value):
    """Check if the value is a real number."""
    return value.is_real

def safe_divide(numerator, denominator):
    """Perform division and handle division by zero."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return float('inf') if numerator > 0 else float('-inf')