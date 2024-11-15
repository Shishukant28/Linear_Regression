from math import exp
from regressions.errors.scalars import three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def logistic_equation(first_constant, second_constant, third_constant, precision = 4):
    """
    Generates a logistic function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Carrying capacity of the resultant logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Growth rate of the resultant logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Value of the sigmoid's midpoint of the resultant logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    evaluation : func
        Function for evaluating a logistic equation when passed any integer or float argument

    See Also
    --------
    :func:`~regressions.analyses.derivatives.logistic.logistic_derivatives`, :func:`~regressions.analyses.integrals.logistic.logistic_integral`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.models.logistic.logistic_model`

    Notes
    -----
    - Standard form of a logistic function: :math:`f(x) = \\frac{a}{1 + \\text{e}^{-b\\cdot(x - c)}}`
    - |logistic_functions|

    Examples
    --------
    Import `logistic_equation` function from `regressions` library
        >>> from regressions.analyses.equations.logistic import logistic_equation
    Create a logistic function with coefficients 2, 3, and 5, then evaluate it at 10
        >>> evaluation_first = logistic_equation(2, 3, 5)
        >>> print(evaluation_first(10))
        2.0
    Create a logistic function with coefficients 100, 5, and 11, then evaluate it at 10
        >>> evaluation_second = logistic_equation(100, 5, 11)
        >>> print(evaluation_second(10))
        0.6693
    Create a logistic function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = logistic_equation(0, 0, 0)
        >>> print(evaluation_zero(10))
        0.0001
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create evaluation
    def logistic_evaluation(variable):
        evaluation = coefficients[0] * (1 + exp(-1 * coefficients[1] * (variable - coefficients[2])))**(-1)
        result = rounded_value(evaluation, precision)
        return result
    return logistic_evaluation