from math import asin, acos, pi
from regressions.errors.scalars import four_scalars, five_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.sort import sorted_list, sorted_strings
from regressions.statistics.rounding import rounded_value, rounded_list
from regressions.vectors.generate import generate_elements
from regressions.vectors.separate import separate_elements
from regressions.analyses.derivatives.sinusoidal import sinusoidal_derivatives

def sinusoidal_roots(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Calculates the roots of a sinusoidal function

    Parameters
    ----------
    first_constant : int or float
        Vertical stretch factor of the original sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Horizontal stretch factor of the original sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Horizontal shift of the original sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    fourth_constant : int or float
        Vertical shift of the original sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    roots : list of float or str
        List of the x-coordinates of the initial x-intercepts within two periods of the original function in float format, along with the general forms in string format that can be used to determine all other x-intercepts by plugging in any integer value for 'k' and evaluating; if the function never crosses the x-axis, then it will return a list of `None`

    See Also
    --------
    :func:`~regressions.analyses.equations.sinusoidal.sinusoidal_equation`, :func:`~regressions.analyses.derivatives.sinusoidal.sinusoidal_derivatives`, :func:`~regressions.analyses.integrals.sinusoidal.sinusoidal_integral`, :func:`~regressions.models.sinusoidal.sinusoidal_model`

    Notes
    -----
    - Standard form of a sinusoidal function: :math:`f(x) = a\\cdot{\\sin(b\\cdot(x - c))} + d`
    - Sinusoidal formula: :math:`x_0 = c + \\frac{1}{b}\\cdot{\\sin^{-1}(-\\frac{d}{a})} + \\frac{2\\pi}{b}\\cdot{k}`

        - :math:`\\text{if} -1 < -\\frac{d}{a} < 0 \\text{ or } 0 < -\\frac{d}{a} < 1, x_1 = c + \\frac{\\pi}{b} - \\frac{1}{b}\\cdot{\\sin^{-1}(-\\frac{d}{a})} + \\frac{2\\pi}{b}\\cdot{k}`
        - :math:`\\text{if} -\\frac{d}{a} = 0, x_1 = c - \\frac{\\pi}{b} + \\frac{2\\pi}{b}\\cdot{k}`
        - :math:`k \\in \\mathbb{Z}`

    Examples
    --------
    Import `sinusoidal_roots` function from `regressions` library
        >>> from regressions.analyses.roots.sinusoidal import sinusoidal_roots
    Calculate the roots of a sinusoidal function with coefficients 2, 3, 5, and 7
        >>> roots_first = sinusoidal_roots(2, 3, 5, 7)
        >>> print(roots_first)
        [None]
    Calculate the roots of a sinusoidal function with coefficients 7, -5, -3, and 2
        >>> roots_second = sinusoidal_roots(7, -5, -3, 2)
        >>> print(roots_second)
        [-8.7128, -7.9686, -7.4562, -6.712, -6.1995, -5.4553, -4.9429, -4.1987, -3.6863, -2.942, '-3.6863 + 1.2566k', '-2.942 + 1.2566k']
    Calculate the roots of a sinusoidal function with all inputs set to 0
        >>> roots_zeroes = sinusoidal_roots(0, 0, 0, 0)
        >>> print(roots_zeroes)
        [-15707.9632, 47123.8899, 109955.743, 172787.596, 235619.4491, '-15707.9632 + 62831.8531k']
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Create list for roots
    roots = []

    # Identify key ratio
    ratio = -1 * coefficients[3] / coefficients[0]

    # Handle no roots
    if ratio > 1 or ratio < -1:
        roots = [None]
    
    # Handle multiple roots
    else:
        # Create intermediary variables
        radians = asin(ratio)
        periodic_radians = radians / coefficients[1]

        # Determine pertinent values
        periodic_unit = 2 * pi / coefficients[1]
        initial_value = coefficients[2] + periodic_radians
        roots = generate_elements(initial_value, periodic_unit, precision)
        
        # Handle roots that bounce on the x-axis
        if ratio == 1 or ratio == -1:
            pass
        
        # Handle roots that cross the x-axis
        else:
            # Determine supplementary values
            alternative_initial_value = coefficients[2] + pi / coefficients[1] - periodic_radians
            generated_elements = generate_elements(alternative_initial_value, periodic_unit, precision)
            
            # Add additional results to roots list
            roots.extend(generated_elements)
    
    # Separate numerical roots, string roots, and None results
    separated_roots = separate_elements(roots)
    numerical_roots = separated_roots['numerical']
    other_roots = separated_roots['other']
    
    # Sort numerical roots
    sorted_roots = sorted_list(numerical_roots)

    # Round numerical roots
    rounded_roots = rounded_list(sorted_roots, precision)
    
    # Sort other_roots
    sorted_other_roots = sorted_strings(other_roots)

    # Combine numerical and non-numerical roots
    result = rounded_roots + sorted_other_roots
    return result

def sinusoidal_roots_first_derivative(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)

    # Generate coefficients of first derivative
    constants = sinusoidal_derivatives(first_constant, second_constant, third_constant, fourth_constant)['first']['constants']

    # Create intermediary variables
    periodic_unit = pi / constants[1]
    initial_value = constants[2] + pi / (2 * constants[1])

    # Generate roots based on criteria
    generated_roots = generate_elements(initial_value, periodic_unit, precision)

    # Return result
    result = generated_roots
    return result

def sinusoidal_roots_second_derivative(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)

    # Generate coefficients of second derivative
    constants = sinusoidal_derivatives(first_constant, second_constant, third_constant, fourth_constant)['second']['constants']

    # Create intermediary variables
    periodic_unit = pi / constants[1]
    initial_value = constants[2]

    # Generate roots based on criteria
    generated_roots = generate_elements(initial_value, periodic_unit, precision)

    # Return result
    result = generated_roots
    return result

def sinusoidal_roots_initial_value(first_constant, second_constant, third_constant, fourth_constant, initial_value, precision = 4):
    # Handle input errors
    five_scalars(first_constant, second_constant, third_constant, fourth_constant, initial_value)
    positive_integer(precision)

    # Determine roots given an initial value
    result = sinusoidal_roots(first_constant, second_constant, third_constant, fourth_constant - initial_value, precision)
    return result

def sinusoidal_roots_derivative_initial_value(first_constant, second_constant, third_constant, fourth_constant, initial_value, precision = 4):
    # Handle input errors
    five_scalars(first_constant, second_constant, third_constant, fourth_constant, initial_value)
    positive_integer(precision)

    # Create intermediary list and list to return
    roots = []
    result = []

    # Identify key ratio
    ratio = initial_value / (first_constant * second_constant)

    # Handle no roots
    if ratio > 1 or ratio < -1:
        result.append(None)
    
    # Handle multiple roots
    else:
        # Handle case in which initial value is zero
        if ratio == 0:
            roots = sinusoidal_roots_first_derivative(first_constant, second_constant, third_constant, fourth_constant, precision)
        
        # Handle general case
        else:
            radians = acos(ratio)
            periodic_radians = radians / second_constant
            periodic_unit = 2 * pi / second_constant
            initial = third_constant + periodic_radians
            roots = generate_elements(initial, periodic_unit, precision)

            # Handle roots that bounce on the x-axis
            if ratio == 1 or ratio == -1:
                pass

            # Handle roots that cross the x-axis
            else:
                alternative_initial = third_constant + periodic_unit - periodic_radians
                generated_elements = generate_elements(alternative_initial, periodic_unit, precision)
                roots.extend(generated_elements)
        
        # Separate numerical roots, string roots, and None results
        separated_roots = separate_elements(roots)
        numerical_roots = separated_roots['numerical']
        other_roots = separated_roots['other']
        
        # Sort numerical roots
        sorted_roots = sorted_list(numerical_roots)

        # Round numerical roots
        rounded_roots = rounded_list(sorted_roots, precision)
        
        # Sort other_roots
        sorted_other_roots = sorted_strings(other_roots)

        # Combine numerical and non-numerical roots
        result.extend(rounded_roots + sorted_other_roots)
    
    # Return result
    return result