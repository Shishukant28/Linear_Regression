from math import log
from regressions.errors.scalars import three_scalars, four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.sort import sorted_list
from regressions.statistics.rounding import rounded_value
from .quadratic import quadratic_roots

def logistic_roots(first_constant, second_constant, third_constant, precision = 4):
    """
    Calculates the roots of a logistic function

    Parameters
    ----------
    first_constant : int or float
        Carrying capacity of the original logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Growth rate of the original logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Value of the sigmoid's midpoint of the original logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the resultant roots

    Raises
    ------
    TypeError
        First three arguments must be integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    roots : list of float
        List of the x-coordinates of all of the x-intercepts of the original function; if the function never crosses the x-axis, then it will return a list of `None`

    See Also
    --------
    :func:`~regressions.analyses.equations.logistic.logistic_equation`, :func:`~regressions.analyses.derivatives.logistic.logistic_derivatives`, :func:`~regressions.analyses.integrals.logistic.logistic_integral`, :func:`~regressions.models.logistic.logistic_model`

    Notes
    -----
    - Standard form of a logistic function: :math:`f(x) = \\frac{a}{1 + \\text{e}^{-b\\cdot(x - c)}}`
    - Logistic formula: :math:`x = \\varnothing`

    Examples
    --------
    Import `logistic_roots` function from `regressions` library
        >>> from regressions.analyses.roots.logistic import logistic_roots
    Calculate the roots of a logistic function with coefficients 2, 3, and 5
        >>> roots_first = logistic_roots(2, 3, 5)
        >>> print(roots_first)
        [None]
    Calculate the roots of a logistic function with coefficients 100, 5, and 11
        >>> roots_second = logistic_roots(100, 5, 11)
        >>> print(roots_second)
        [None]
    Calculate the roots of a logistic function with all inputs set to 0
        >>> roots_zeroes = logistic_roots(0, 0, 0)
        >>> print(roots_zeroes)
        [None]
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create list to return
    result = []

    # Determine root
    root = None

    # Return result
    result.append(root)
    return result

def logistic_roots_first_derivative(first_constant, second_constant, third_constant, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    
    # Create list to return
    result = []

    # Determine root of first derivative
    root = None

    # Return result
    result.append(root)
    return result

def logistic_roots_second_derivative(first_constant, second_constant, third_constant, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine root of second derivative
    root = rounded_value(third_constant)

    # Return root
    result.append(root)
    return result

def logistic_roots_initial_value(first_constant, second_constant, third_constant, initial_value, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, initial_value)
    positive_integer(precision)

    # Create list to return
    result = []
    
    # Create pivot variable
    log_argument = first_constant / initial_value - 1

    # Circumvent logarithm of zero
    if log_argument == 0:
        log_argument = 10**(-precision)

    # Create intermediary variables
    numerator = log(abs(log_argument))
    denominator = second_constant
    ratio = numerator / denominator

    # Determine root given an initial value
    root = third_constant - ratio

    # Round root
    rounded_root = rounded_value(root, precision)

    # Return result
    result.append(rounded_root)
    return result

def logistic_roots_derivative_initial_value(first_constant, second_constant, third_constant, initial_value, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, initial_value)
    positive_integer(precision)

    # Create intermediary list and list to return
    roots = []
    result = []

    # Determine quadratic roots of derivative given an initial value
    intermediary_roots = quadratic_roots(initial_value, 2 * initial_value - first_constant * second_constant, initial_value, precision)

    # Handle no roots
    if intermediary_roots[0] == None:
        roots.append(None)
    
    # Convert quadratic roots using logarithms
    else:
        for intermediary in intermediary_roots:
            if intermediary == 0:
                intermediary = 10**(-precision)
            root = third_constant - log(abs(intermediary)) / second_constant
            rounded_root = rounded_value(root, precision)
            roots.append(rounded_root)
    
    # Sort roots
    sorted_roots = sorted_list(roots)

    # Return result
    result.extend(sorted_roots)
    return result