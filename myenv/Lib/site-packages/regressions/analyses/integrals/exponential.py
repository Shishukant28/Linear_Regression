from math import log
from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def exponential_integral(first_constant, second_constant, precision = 4):
    """
    Generates the integral of an exponential function

    Parameters
    ----------
    first_constant : int or float
        Constant multiple of the original exponential function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Base rate of variable of the original exponential function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001); if one, it will be converted to a small, near-one decimal value (e.g., 1.0001)
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
    integral['constants'] : list of float
        Coefficients of the resultant integral
    integral['evaluation'] : func
        Function for evaluating the resultant integral at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.exponential.exponential_equation`, :func:`~regressions.analyses.derivatives.exponential.exponential_derivatives`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.models.exponential.exponential_model`

    Notes
    -----
    - Standard form of an exponential function: :math:`f(x) = a\\cdot{b^x}`
    - Integral of an exponential function: :math:`F(x) = \\frac{a}{\\ln{b}}\\cdot{b^x}`
    - |indefinite_integral|
    - |integration_formulas|

    Examples
    --------
    Import `exponential_integral` function from `regressions` library
        >>> from regressions.analyses.integrals.exponential import exponential_integral
    Generate the integral of an exponential function with coefficients 2 and 3, then display its coefficients
        >>> integral_constants = exponential_integral(2, 3)
        >>> print(integral_constants['constants'])
        [1.8205, 3.0]
    Generate the integral of an exponential function with coefficients -2 and 3, then evaluate its integral at 10
        >>> integral_evaluation = exponential_integral(-2, 3)
        >>> print(integral_evaluation['evaluation'](10))
        -107498.7045
    Generate the integral of an exponential function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = exponential_integral(0, 0)
        >>> print(integral_zeroes['constants'])
        [-0.0001, 0.0001]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Circumvent division by zero
    if coefficients[1] == 1:
        coefficients[1] = 1 + 10**(-precision)
    
    # Create constants
    integral_coefficients = [coefficients[0] / log(abs(coefficients[1])), coefficients[1]]
    constants = rounded_list(integral_coefficients, precision)

    # Create evaluation
    def exponential_evaluation(variable):
        evaluation = constants[0] * constants[1]**variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': exponential_evaluation
    }
    return results