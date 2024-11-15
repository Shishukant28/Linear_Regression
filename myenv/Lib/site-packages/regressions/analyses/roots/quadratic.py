from regressions.errors.scalars import three_scalars, four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.sort import sorted_list
from regressions.statistics.rounding import rounded_value, rounded_list
from regressions.analyses.derivatives.quadratic import quadratic_derivatives
from .linear import linear_roots

def quadratic_roots(first_constant, second_constant, third_constant, precision = 4):
    """
    Calculates the roots of a quadratic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the quadratic term of the original quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the linear term of the original quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Coefficient of the constant term of the original quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    :func:`~regressions.analyses.equations.quadratic.quadratic_equation`, :func:`~regressions.analyses.derivatives.quadratic.quadratic_derivatives`, :func:`~regressions.analyses.integrals.quadratic.quadratic_integral`, :func:`~regressions.models.quadratic.quadratic_model`

    Notes
    -----
    - Standard form of a quadratic function: :math:`f(x) = a\\cdot{x^2} + b\\cdot{x} + c`
    - Quadratic formula: :math:`x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}`
    - |quadratic_formula|

    Examples
    --------
    Import `quadratic_roots` function from `regressions` library
        >>> from regressions.analyses.roots.quadratic import quadratic_roots
    Calculate the roots of a quadratic function with coefficients 2, 7, and 5
        >>> roots_first = quadratic_roots(2, 7, 5)
        >>> print(roots_first)
        [-2.5, -1.0]
    Calculate the roots of a quadratic function with coefficients 2, -5, and 3
        >>> roots_second = quadratic_roots(2, -5, 3)
        >>> print(roots_second)
        [1.0, 1.5]
    Calculate the roots of a quadratic function with all inputs set to 0
        >>> roots_zeroes = quadratic_roots(0, 0, 0)
        >>> print(roots_zeroes)
        [None]
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create intermediary list and list to return
    roots = []
    result = []

    # Create intermediary variable
    discriminant = coefficients[1]**2 - 4 * coefficients[0] * coefficients[2]

    # Create roots
    first_root = (-1 * coefficients[1] + discriminant**(1/2)) / (2 * coefficients[0])
    second_root = (-1 * coefficients[1] - discriminant**(1/2)) / (2 * coefficients[0])

    # Eliminate duplicate roots
    if first_root == second_root:
        roots.append(first_root)
    
    # Eliminate complex roots
    else:
        if not isinstance(first_root, complex):
            roots.append(first_root)
        if not isinstance(second_root, complex):
            roots.append(second_root)
    
    # Handle no roots
    if not roots:
        roots.append(None)
    
    # Sort roots
    sorted_roots = sorted_list(roots)
    
    # Round roots
    rounded_roots = rounded_list(sorted_roots, precision)

    # Return result
    result.extend(rounded_roots)
    return result

def quadratic_roots_first_derivative(first_constant, second_constant, third_constant, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)

    # Generate coefficients of first derivative
    constants = quadratic_derivatives(first_constant, second_constant, third_constant)['first']['constants']

    # Determine roots of first derivative
    result = linear_roots(*constants, precision)
    return result

def quadratic_roots_second_derivative(first_constant, second_constant, third_constant, precision = 4):
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    
    # Create list to return
    result = []
    
    # Determine root
    root = None

    # Return result
    result.append(root)
    return result

def quadratic_roots_initial_value(first_constant, second_constant, third_constant, initial_value, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, initial_value)
    positive_integer(precision)

    # Determine roots given an initial value
    result = quadratic_roots(first_constant, second_constant, third_constant - initial_value, precision)
    return result

def quadratic_roots_derivative_initial_value(first_constant, second_constant, third_constant, initial_value, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, initial_value)
    positive_integer(precision)

    # Determine roots of derivative given an initial value
    result = linear_roots(2 * first_constant, second_constant - initial_value, precision)
    return result