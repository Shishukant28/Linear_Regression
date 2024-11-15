from regressions.errors.scalars import three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def quadratic_integral(first_constant, second_constant, third_constant, precision = 4):
    """
    Generates the integral of a quadratic function

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
    integral['constants'] : list of float
        Coefficients of the resultant integral
    integral['evaluation'] : func
        Function for evaluating the resultant integral at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.quadratic.quadratic_equation`, :func:`~regressions.analyses.derivatives.quadratic.quadratic_derivatives`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.models.quadratic.quadratic_model`

    Notes
    -----
    - Standard form of a quadratic function: :math:`f(x) = a\\cdot{x^2} + b\\cdot{x} + c`
    - Integral of a quadratic function: :math:`F(x) = \\frac{a}{3}\\cdot{x^3} + \\frac{b}{2}\\cdot{x^2} + c\\cdot{x}`
    - |indefinite_integral|
    - |integration_formulas|

    Examples
    --------
    Import `quadratic_integral` function from `regressions` library
        >>> from regressions.analyses.integrals.quadratic import quadratic_integral
    Generate the integral of a quadratic function with coefficients 2, 3, and 5, then display its coefficients
        >>> integral_constants = quadratic_integral(2, 3, 5)
        >>> print(integral_constants['constants'])
        [0.6667, 1.5, 5.0]
    Generate the integral of a quadratic function with coefficients 7, -5, and 3, then evaluate its integral at 10
        >>> integral_evaluation = quadratic_integral(7, -5, 3)
        >>> print(integral_evaluation['evaluation'](10))
        2113.3
    Generate the integral of a quadratic function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = quadratic_integral(0, 0, 0)
        >>> print(integral_zeroes['constants'])
        [0.0001, 0.0001, 0.0001]
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create constants
    integral_coefficients = [(1/3) * coefficients[0], (1/2) * coefficients[1], coefficients[2]]
    constants = rounded_list(integral_coefficients, precision)

    # Create evaluation
    def quadratic_evaluation(variable):
        evaluation = constants[0] * variable**3 + constants[1] * variable**2 + constants[2] * variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': quadratic_evaluation
    }
    return results