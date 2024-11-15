from math import cos
from regressions.errors.scalars import four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def sinusoidal_integral(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Generates the integral of a sinusoidal function

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
    integral['constants'] : list of float
        Coefficients of the resultant integral
    integral['evaluation'] : func
        Function for evaluating the resultant integral at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.sinusoidal.sinusoidal_equation`, :func:`~regressions.analyses.derivatives.sinusoidal.sinusoidal_derivatives`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`, :func:`~regressions.models.sinusoidal.sinusoidal_model`

    Notes
    -----
    - Standard form of a sinusoidal function: :math:`f(x) = a\\cdot{\\sin(b\\cdot(x - c))} + d`
    - Integral of a sinusoidal function: :math:`F(x) = -\\frac{a}{b}\\cdot{\\cos(b\\cdot(x - c))} + d\\cdot{x}`
    - |indefinite_integral|
    - |integration_formulas|
    - |substitution_rule|

    Examples
    --------
    Import `sinusoidal_integral` function from `regressions` library
        >>> from regressions.analyses.integrals.sinusoidal import sinusoidal_integral
    Generate the integral of a sinusoidal function with coefficients 2, 3, 5, and 7, then display its coefficients
        >>> integral_constants = sinusoidal_integral(2, 3, 5, 7)
        >>> print(integral_constants['constants'])
        [-0.6667, 3.0, 5.0, 7.0]
    Generate the integral of a sinusoidal function with coefficients 7, -5, -3, and 2, then evaluate its integral at 10
        >>> integral_evaluation = sinusoidal_integral(7, -5, -3, 2)
        >>> print(integral_evaluation['evaluation'](10))
        19.2126
    Generate the integral of a sinusoidal function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = sinusoidal_integral(0, 0, 0, 0)
        >>> print(integral_zeroes['constants'])
        [-1.0, 0.0001, 0.0001, 0.0001]
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)
    
    # Create constants
    integral_coefficients = [-1 * coefficients[0] / coefficients[1], coefficients[1], coefficients[2], coefficients[3]]
    constants = rounded_list(integral_coefficients, precision)

    # Create evaluation
    def sinusoidal_evaluation(variable):
        evaluation = constants[0] * cos(constants[1] * (variable - constants[2])) + constants[3] * variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': sinusoidal_evaluation
    }
    return results