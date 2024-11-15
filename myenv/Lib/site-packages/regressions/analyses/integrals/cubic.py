from regressions.errors.scalars import four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def cubic_integral(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Generates the integral of a cubic function

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
    integral['constants'] : list of float
        Coefficients of the resultant integral
    integral['evaluation'] : func
        Function for evaluating the resultant integral at any float or integer argument
    
    See Also
    --------
    :func:`~regressions.analyses.equations.cubic.cubic_equation`, :func:`~regressions.analyses.derivatives.cubic.cubic_derivatives`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.models.cubic.cubic_model`

    Notes
    -----
    - Standard form of a cubic function: :math:`f(x) = a\\cdot{x^3} + b\\cdot{x^2} + c\\cdot{x} + d`
    - Integral of a cubic function: :math:`F(x) = \\frac{a}{4}\\cdot{x^4} + \\frac{b}{3}\\cdot{x^3} + \\frac{c}{2}\\cdot{x^2} + d\\cdot{x}`
    - |indefinite_integral|
    - |integration_formulas|

    Examples
    --------
    Import `cubic_integral` function from `regressions` library
        >>> from regressions.analyses.integrals.cubic import cubic_integral
    Generate the integral of a cubic function with coefficients 2, 3, 5, and 7, then display its coefficients
        >>> integral_constants = cubic_integral(2, 3, 5, 7)
        >>> print(integral_constants['constants'])
        [0.5, 1.0, 2.5, 7.0]
    Generate the integral of a cubic function with coefficients 7, -5, -3, and 2, then evaluate its integral at 10
        >>> integral_evaluation = cubic_integral(7, -5, -3, 2)
        >>> print(integral_evaluation['evaluation'](10))
        15703.3
    Generate the integral of a cubic function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = cubic_integral(0, 0, 0, 0)
        >>> print(integral_zeroes['constants'])
        [0.0001, 0.0001, 0.0001, 0.0001]
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Generate constants
    integral_coefficients = [(1/4) * coefficients[0], (1/3) * coefficients[1], (1/2) * coefficients[2], coefficients[3]]
    constants = rounded_list(integral_coefficients, precision)

    # Create evaluation
    def cubic_evaluation(variable):
        evaluation = constants[0] * variable**4 + constants[1] * variable**3 + constants[2] * variable**2 + constants[3] * variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': cubic_evaluation
    }
    return results