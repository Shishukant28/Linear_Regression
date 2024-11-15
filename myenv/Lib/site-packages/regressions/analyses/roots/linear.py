from regressions.errors.scalars import two_scalars, three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def linear_roots(first_constant, second_constant, precision = 4):
    """
    Calculates the roots of a linear function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the linear term of the original linear function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the original linear function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    :func:`~regressions.analyses.equations.linear.linear_equation`, :func:`~regressions.analyses.derivatives.linear.linear_derivatives`, :func:`~regressions.analyses.integrals.linear.linear_integral`, :func:`~regressions.models.linear.linear_model`

    Notes
    -----
    - Standard form of a linear function: :math:`f(x) = a\\cdot{x} + b`
    - Linear formula: :math:`x = -\\frac{b}{a}`
    - |linear_formula|

    Examples
    --------
    Import `linear_roots` function from `regressions` library
        >>> from regressions.analyses.roots.linear import linear_roots
    Calculate the roots of a linear function with coefficients 2 and 3
        >>> roots_first = linear_roots(2, 3)
        >>> print(roots_first)
        [-1.5]
    Calculate the roots of a linear function with coefficients -2 and 3
        >>> roots_second = linear_roots(-2, 3)
        >>> print(roots_second)
        [1.5]
    Calculate the roots of a linear function with all inputs set to 0
        >>> roots_zeroes = linear_roots(0, 0)
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
    root = -1 * coefficients[1] / coefficients[0]

    # Round root
    rounded_root = rounded_value(root, precision)

    # Return result
    result.append(rounded_root)
    return result

def linear_roots_first_derivative(first_constant, second_constant, precision = 4):
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine root of first derivative
    root = None

    # Return result
    result.append(root)
    return result

def linear_roots_second_derivative(first_constant, second_constant, precision = 4):
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine root of second derivative
    root = None

    # Return result
    result.append(root)
    return result

def linear_roots_initial_value(first_constant, second_constant, initial_value, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, initial_value)
    positive_integer(precision)

    # Determine roots given an initial value
    result = linear_roots(first_constant, second_constant - initial_value, precision)
    return result

def linear_roots_derivative_initial_value(first_constant, second_constant, initial_value, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, initial_value)
    positive_integer(precision)

    # Create list to return
    result = []

    # Handle general case
    if initial_value == first_constant:
        result.append('All')
    
    # Handle exception
    else:
        result.append(None)
    
    # Return result
    return result