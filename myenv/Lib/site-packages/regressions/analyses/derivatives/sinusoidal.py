from math import sin, cos
from regressions.errors.scalars import four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def sinusoidal_derivatives(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Calculates the first and second derivatives of a sinusoidal function

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
    derivatives['first']['constants'] : list of float
        Coefficients of the resultant first derivative
    derivatives['first']['evaluation'] : func
        Function for evaluating the resultant first derivative at any float or integer argument
    derivatives['second']['constants'] : list of float
        Coefficients of the resultant second derivative
    derivatives['second']['evaluation'] : func
        Function for evaluating the resultant second derivative at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.sinusoidal.sinusoidal_equation`, :func:`~regressions.analyses.integrals.sinusoidal.sinusoidal_integral`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`, :func:`~regressions.models.sinusoidal.sinusoidal_model`

    Notes
    -----
    - Standard form of a sinusoidal function: :math:`f(x) = a\\cdot{\\sin(b\\cdot(x - c))} + d`
    - First derivative of a sinusoidal function: :math:`f'(x) = ab\\cdot{\\cos(b\\cdot(x - c))}`
    - Second derivative of a sinusoidal function: :math:`f''(x) = -ab^2\\cdot{\\sin(b\\cdot(x - c))}`
    - |differentiation_formulas|
    - |chain_rule|
    - |trigonometric|

    Examples
    --------
    Import `sinusoidal_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.sinusoidal import sinusoidal_derivatives
    Generate the derivatives of a sinusoidal function with coefficients 2, 3, 5, and 7, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = sinusoidal_derivatives(2, 3, 5, 7)
        >>> print(derivatives_constants['first']['constants'])
        [6.0, 3.0, 5.0]
        >>> print(derivatives_constants['second']['constants'])
        [-18.0, 3.0, 5.0]
    Generate the derivatives of a sinusoidal function with coefficients 7, -5, -3, and 2, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = sinusoidal_derivatives(7, -5, -3, 2)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        19.6859
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        144.695
    Generate the derivatives of a sinusoidal function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = sinusoidal_derivatives(0, 0, 0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [0.0001, 0.0001, 0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [-0.0001, 0.0001, 0.0001]
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Create first derivative
    first_coefficients = [coefficients[0] * coefficients[1], coefficients[1], coefficients[2]]
    first_constants = rounded_list(first_coefficients, precision)
    def first_derivative(variable):
        evaluation = first_constants[0] * cos(first_constants[1] * (variable - first_constants[2]))
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_coefficients = [-1 * first_constants[0] * first_constants[1], first_constants[1], first_constants[2]]
    second_constants = rounded_list(second_coefficients, precision)
    def second_derivative(variable):
        evaluation = second_constants[0] * sin(second_constants[1] * (variable - second_constants[2]))
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    second_dictionary = {
        'constants': second_constants,
        'evaluation': second_derivative
    }

    # Package both derivatives in single dictionary
    results = {
        'first': first_dictionary,
        'second': second_dictionary
    }
    return results