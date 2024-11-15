from math import exp
from regressions.errors.scalars import two_scalars, three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value
from .hyperbolic import hyperbolic_roots

def logarithmic_roots(first_constant, second_constant, precision = 4):
    """
    Calculates the roots of a logarithmic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the logarithmic term of the original logarithmic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the original logarithmic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        List of the x-coordinates of all of the x-intercepts of the original function

    See Also
    --------
    :func:`~regressions.analyses.equations.logarithmic.logarithmic_equation`, :func:`~regressions.analyses.derivatives.logarithmic.logarithmic_derivatives`, :func:`~regressions.analyses.integrals.logarithmic.logarithmic_integral`, :func:`~regressions.models.logarithmic.logarithmic_model`

    Notes
    -----
    - Standard form of a logarithmic function: :math:`f(x) = a\\cdot{\\ln{x}} + b`
    - Logarithmic formula: :math:`x = \\text{e}^{-\\frac{b}{a}}`

    Examples
    --------
    Import `logarithmic_roots` function from `regressions` library
        >>> from regressions.analyses.roots.logarithmic import logarithmic_roots
    Calculate the roots of a logarithmic function with coefficients 2 and 3
        >>> roots_first = logarithmic_roots(2, 3)
        >>> print(roots_first)
        [0.2231]
    Calculate the roots of a logarithmic function with coefficients -2 and 3
        >>> roots_second = logarithmic_roots(-2, 3)
        >>> print(roots_second)
        [4.4817]
    Calculate the roots of a logarithmic function with all inputs set to 0
        >>> roots_zeroes = logarithmic_roots(0, 0)
        >>> print(roots_zeroes)
        [0.3679]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create list to return
    result = []
    
    # Determine root
    root = exp(-1 * coefficients[1] / coefficients[0])

    # Round root
    rounded_root = rounded_value(root, precision)

    # Return result
    result.append(rounded_root)
    return result

def logarithmic_roots_first_derivative(first_constant, second_constant, precision = 4):
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)

    # Create list to return
    result = []
    
    # Determine root
    root = None

    # Return result
    result.append(root)
    return result

def logarithmic_roots_second_derivative(first_constant, second_constant, precision = 4):
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)

    # Create list to return
    result = []
    
    # Determine root
    root = None

    # Return result
    result.append(root)
    return result

def logarithmic_roots_initial_value(first_constant, second_constant, initial_value, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, initial_value)
    positive_integer(precision)

    # Determine roots given an initial value
    result = logarithmic_roots(first_constant, second_constant - initial_value, precision)
    return result

def logarithmic_roots_derivative_initial_value(first_constant, second_constant, initial_value, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, initial_value)
    positive_integer(precision)

    # Determine roots of derivative given an initial value
    result = hyperbolic_roots(first_constant, -1 * initial_value, precision)
    return result