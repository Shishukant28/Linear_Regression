from regressions.errors.scalars import four_scalars, five_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.sort import sorted_list
from regressions.statistics.rounding import rounded_value, rounded_list
from regressions.analyses.derivatives.cubic import cubic_derivatives
from .quadratic import quadratic_roots
from .linear import linear_roots

def cubic_roots(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Calculates the roots of a cubic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the cubic term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the quadratic term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Coefficient of the linear term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    fourth_constant : int or float
        Coefficient of the constant term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the resultant roots

    Raises
    ------
    TypeError
        First four arguments must be integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    roots : list of float
        List of the x-coordinates of all of the x-intercepts of the original function

    See Also
    --------
    :func:`~regressions.analyses.equations.cubic.cubic_equation`, :func:`~regressions.analyses.derivatives.cubic.cubic_derivatives`, :func:`~regressions.analyses.integrals.cubic.cubic_integral`, :func:`~regressions.models.cubic.cubic_model`

    Notes
    -----
    - Standard form of a cubic function: :math:`f(x) = a\\cdot{x^3} + b\\cdot{x^2} + c\\cdot{x} + d`
    - Cubic formula: :math:`x_k = -\\frac{1}{3a}\\cdot(b + \\xi^k\\cdot{\\eta} + \\frac{\\Delta_0}{\\xi^k\\cdot{\\eta}})`

        - :math:`\\Delta_0 = b^2 - 3ac`
        - :math:`\\Delta_1 = 2b^3 - 9abc +27a^2d`
        - :math:`\\xi = \\frac{-1 + \\sqrt{-3}}{2}`
        - :math:`\\eta = \\sqrt[3]{\\frac{\\Delta_1 \\pm \\sqrt{\\Delta_1^2 - 4\\Delta_0^3}}{2}}`
        - :math:`k \\in \\{ 0, 1, 2 \\}`
    
    - |cubic_formula|

    Examples
    --------
    Import `cubic_roots` function from `regressions` library
        >>> from regressions.analyses.roots.cubic import cubic_roots
    Calculate the roots of a cubic function with coefficients 2, 3, 5, and 7
        >>> roots_first = cubic_roots(2, 3, 5, 7)
        >>> print(roots_first)
        [-1.4455]
    Calculate the roots of a cubic function with coefficients 7, -5, -3, and 2
        >>> roots_second = cubic_roots(7, -5, -3, 2)
        >>> print(roots_second)
        [-0.6431, 0.551, 0.8064]
    Calculate the roots of a cubic function with all inputs set to 0
        >>> roots_zeroes = cubic_roots(0, 0, 0, 0)
        >>> print(roots_zeroes)
        [-1.0]
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Create intermediary variables
    xi = (-1 + (-3)**(1/2)) / 2
    delta_first = coefficients[1]**2 - 3 * coefficients[0] * coefficients[2]
    delta_second = 2 * coefficients[1]**3 - 9 * coefficients[0] * coefficients[1] * coefficients[2] + 27 * coefficients[0]**2 * coefficients[3]
    discriminant = delta_second**2 - 4 * delta_first**3
    eta_first = ((delta_second + discriminant**(1/2)) / 2)**(1/3)
    eta_second = ((delta_second - discriminant**(1/2)) / 2)**(1/3)
    eta = 0
    if eta_first == 0:
        eta = eta_second
    else:
        eta = eta_first
    
    # Create roots
    roots = []
    first_root = (-1 / (3 * coefficients[0])) * (coefficients[1] + eta * xi**0 + delta_first / (eta * xi**0))
    second_root = (-1 / (3 * coefficients[0])) * (coefficients[1] + eta * xi**1 + delta_first / (eta * xi**1))
    third_root = (-1 / (3 * coefficients[0])) * (coefficients[1] + eta * xi**2 + delta_first / (eta * xi**2))

    # Identify real and imaginary components of complex roots
    first_real = first_root.real
    second_real = second_root.real
    third_real = third_root.real
    first_imag = first_root.imag
    second_imag = second_root.imag
    third_imag = third_root.imag

    # Determine magnitudes of imaginary components
    size_first_imag = (first_imag**2)**(1/2)
    size_second_imag = (second_imag**2)**(1/2)
    size_third_imag = (third_imag**2)**(1/2)

    # Eliminate roots with large imaginary components
    if size_first_imag < 10**(-precision):
        first_root = first_real
        roots.append(first_root)
    if size_second_imag < 10**(-precision):
        second_root = second_real
        roots.append(second_root)
    if size_third_imag < 10**(-precision):
        third_root = third_real
        roots.append(third_root)
    
    # Eliminate duplicate roots
    unique_roots = list(set(roots))
    
    # Sort unique roots
    sorted_roots = sorted_list(unique_roots)

    # Round roots
    rounded_roots = rounded_list(sorted_roots, precision)

    # Return result
    result = rounded_roots
    return result

def cubic_roots_first_derivative(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)

    # Generate coefficients of first derivative
    constants = cubic_derivatives(first_constant, second_constant, third_constant, fourth_constant)['first']['constants']

    # Determine roots of first derivative
    result = quadratic_roots(*constants, precision)
    return result

def cubic_roots_second_derivative(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)

    # Generate coefficients of second derivative
    constants = cubic_derivatives(first_constant, second_constant, third_constant, fourth_constant)['second']['constants']

    # Determine roots of second derivative
    result = linear_roots(*constants, precision)
    return result

def cubic_roots_initial_value(first_constant, second_constant, third_constant, fourth_constant, initial_value, precision = 4):
    # Handle input errors
    five_scalars(first_constant, second_constant, third_constant, fourth_constant, initial_value)
    positive_integer(precision)

    # Determine roots given an initial value
    result = cubic_roots(first_constant, second_constant, third_constant, fourth_constant - initial_value, precision)
    return result

def cubic_roots_derivative_initial_value(first_constant, second_constant, third_constant, fourth_constant, initial_value, precision = 4):
    # Handle input errors
    five_scalars(first_constant, second_constant, third_constant, fourth_constant, initial_value)
    positive_integer(precision)

    # Determine roots of derivative given an initial value
    result = quadratic_roots(3 * first_constant, 2 * second_constant, third_constant - initial_value, precision)
    return result