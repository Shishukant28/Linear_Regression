from math import log
from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def hyperbolic_integral(first_constant, second_constant, precision = 4):
    """
    Generates the integral of a hyperbolic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the reciprocal variable of the original hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the original hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating the resultant integral at any float or integer argument; if zero inputted as argument, it will be converted to a small, non-zero decimal value (e.g., 0.0001)

    See Also
    --------
    :func:`~regressions.analyses.equations.hyperbolic.hyperbolic_equation`, :func:`~regressions.analyses.derivatives.hyperbolic.hyperbolic_derivatives`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.models.hyperbolic.hyperbolic_model`

    Notes
    -----
    - Standard form of a hyperbolic function: :math:`f(x) = a\\cdot{\\frac{1}{x}} + b`
    - Integral of a hyperbolic function: :math:`F(x) = a\\cdot{\\ln|x|} + b\\cdot{x}`
    - |indefinite_integral|
    - |integration_formulas|

    Examples
    --------
    Import `sinusoidal_hyperbolic` function from `regressions` library
        >>> from regressions.analyses.hyperbolics.sinusoidal import sinusoidal_hyperbolic
    Generate the integral of a hyperbolic function with coefficients 2 and 3, then display its coefficients
        >>> integral_constants = hyperbolic_integral(2, 3)
        >>> print(integral_constants['constants'])
        [2.0, 3.0]
    Generate the integral of a hyperbolic function with coefficients -2 and 3, then evaluate its integral at 10
        >>> integral_evaluation = hyperbolic_integral(-2, 3)
        >>> print(integral_evaluation['evaluation'](10))
        25.3948
    Generate the integral of a hyperbolic function with all inputs set to 0, then display its coefficients
        >>> integral_zeroes = hyperbolic_integral(0, 0)
        >>> print(integral_zeroes['constants'])
        [0.0001, 0.0001]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create constants
    constants = [coefficients[0], coefficients[1]]

    # Create evaluation
    def hyperbolic_evaluation(variable):
        # Circumvent logarithm of zero
        if variable == 0:
            variable = 10**(-precision)
        evaluation = constants[0] * log(abs(variable)) + constants[1] * variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    
    # Package constants and evaluation in single dictionary
    results = {
        'constants': constants,
        'evaluation': hyperbolic_evaluation
    }
    return results