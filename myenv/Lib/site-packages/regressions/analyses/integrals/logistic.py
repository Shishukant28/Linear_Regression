from math import exp, log
from regressions.errors.scalars import three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def logistic_integral(first_constant, second_constant, third_constant, precision = 4):
    """
    Generates the integral of a logistic function

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
    integral['constants'] : list of float
        Coefficients of the resultant integral
    integral['evaluation'] : func
        Function for evaluating the resultant integral at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.logistic.logistic_equation`, :func:`~regressions.analyses.derivatives.logistic.logistic_derivatives`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.models.logistic.logistic_model`

    Notes
    -----
    - Standard form of a logistic function: :math:`f(x) = \\frac{a}{1 + \\text{e}^{-b\\cdot(x - c)}}`
    - Integral of a logistic function: :math:`F(x) = \\frac{a}{b}\\cdot{\\ln|\\text{e}^{b\\cdot(x - c)} + 1|}`
    - |indefinite_integral|
    - |integration_formulas|
    - |substitution_rule|

    Examples
    --------
    Import `logistic_integral` function from `regressions` library
        >>> from regressions.analyses.integrals.logistic import logistic_integral
    Generate the integral of a logistic function with coefficients 2, 3, and 5, then display its coefficients
        >>> integral_constants = logistic_integral(2, 3, 5)
        >>> print(integral_constants['constants'])
        [0.6667, 3.0, 5.0]
    Generate the integral of a logistic function with coefficients 100, 5, and 11, then evaluate its integral at 10
        >>> integral_evaluation = logistic_integral(100, 5, 11)
        >>> print(integral_evaluation['evaluation'](10))
        0.1343
    Generate the integral of a logistic function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = logistic_integral(0, 0, 0)
        >>> print(integral_zeroes['constants'])
        [1.0, 0.0001, 0.0001]
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create constants
    integral_coefficients = [coefficients[0] / coefficients[1], coefficients[1], coefficients[2]]
    constants = rounded_list(integral_coefficients, precision)

    # Create evaluation
    def logistic_evaluation(variable):
        evaluation = constants[0] * log(abs(exp(constants[1] * (variable - constants[2])) + 1))
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': logistic_evaluation
    }
    return results