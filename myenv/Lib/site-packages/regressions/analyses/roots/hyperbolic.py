from regressions.errors.scalars import two_scalars, three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def hyperbolic_roots(first_constant, second_constant, precision = 4):
    """
    Calculates the roots of a hyperbolic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the reciprocal variable of the original hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the original hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the resultant roots

    Raises
    ------
    TypeError
        First two arguments must be integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    roots : list of float
        List of the x-coordinates of all of the x-intercepts of the original function; if the function never crosses the x-axis, then it will return a list of `None`

    See Also
    --------
    :func:`~regressions.analyses.equations.hyperbolic.hyperbolic_equation`, :func:`~regressions.analyses.derivatives.hyperbolic.hyperbolic_derivatives`, :func:`~regressions.analyses.integrals.hyperbolic.hyperbolic_integral`, :func:`~regressions.models.hyperbolic.hyperbolic_model`

    Notes
    -----
    - Standard form of a hyperbolic function: :math:`f(x) = a\\cdot{\\frac{1}{x}} + b`
    - Hyperbolic formula: :math:`x = -\\frac{a}{b}`

    Examples
    --------
    Import `hyperbolic_roots` function from `regressions` library
        >>> from regressions.analyses.roots.hyperbolic import hyperbolic_roots
    Calculate the roots of a hyperbolic function with coefficients 2 and 3
        >>> roots_first = hyperbolic_roots(2, 3)
        >>> print(roots_first)
        [-0.6667]
    Calculate the roots of a hyperbolic function with coefficients -2 and 3
        >>> roots_second = hyperbolic_roots(-2, 3)
        >>> print(roots_second)
        [0.6667]
    Calculate the roots of a hyperbolic function with all inputs set to 0
        >>> roots_zeroes = hyperbolic_roots(0, 0)
        >>> print(roots_zeroes)
        [-1.0]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create list to return
    result = []
    
    # Determine root
    root = -1 * coefficients[0] / coefficients[1]

    # Round root
    rounded_root = rounded_value(root, precision)

    # Return result
    result.append(rounded_root)
    return result

def hyperbolic_roots_first_derivative(first_constant, second_constant, precision = 4):
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine root of first derivative
    root = 0.0

    # Return result
    result.append(root)
    return result

def hyperbolic_roots_second_derivative(first_constant, second_constant, precision = 4):
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine root of second derivative
    root = 0.0

    # Return result
    result.append(root)
    return result

def hyperbolic_roots_initial_value(first_constant, second_constant, initial_value, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, initial_value)
    positive_integer(precision)

    # Determine roots given an initial value
    result = hyperbolic_roots(first_constant, second_constant - initial_value, precision)
    return result

def hyperbolic_roots_derivative_initial_value(first_constant, second_constant, initial_value, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, initial_value)
    positive_integer(precision)

    # Create list to return
    result = []

    # Create intermediary variable
    ratio = -1 * first_constant / initial_value

    # Handle no roots
    if ratio < 0:
        result.append(None)
    
    # Determine roots of derivative given an initial value
    else:
        radical = ratio**(1/2)
        rounded_radical = rounded_value(radical, precision)
        result.append(rounded_radical)
    return result