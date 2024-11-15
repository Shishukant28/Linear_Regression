from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def linear_integral(first_constant, second_constant, precision = 4):
    """
    Generates the integral of a linear function

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
    integral['constants'] : list of float
        Coefficients of the resultant integral
    integral['evaluation'] : func
        Function for evaluating the resultant integral at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.linear.linear_equation`, :func:`~regressions.analyses.derivatives.linear.linear_derivatives`, :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.models.linear.linear_model`

    Notes
    -----
    - Standard form of a linear function: :math:`f(x) = a\\cdot{x} + b`
    - Integral of a linear function: :math:`F(x) = \\frac{a}{2}\\cdot{x^2} + b\\cdot{x}`
    - |indefinite_integral|
    - |integration_formulas|

    Examples
    --------
    Import `linear_integral` function from `regressions` library
        >>> from regressions.analyses.integrals.linear import linear_integral
    Generate the integral of a linear function with coefficients 2 and 3, then display its coefficients
        >>> integral_constants = linear_integral(2, 3)
        >>> print(integral_constants['constants'])
        [1.0, 3.0]
    Generate the integral of a linear function with coefficients -2 and 3, then evaluate its integral at 10
        >>> integral_evaluation = linear_integral(-2, 3)
        >>> print(integral_evaluation['evaluation'](10))
        -70.0
    Generate the integral of a linear function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = linear_integral(0, 0)
        >>> print(integral_zeroes['constants'])
        [0.0001, 0.0001]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create constants
    integral_coefficients = [(1/2) * coefficients[0], coefficients[1]]
    constants = rounded_list(integral_coefficients, precision)

    # Create evaluation
    def linear_evaluation(variable):
        evaluation = constants[0] * variable**2 + constants[1] * variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': linear_evaluation
    }
    return results