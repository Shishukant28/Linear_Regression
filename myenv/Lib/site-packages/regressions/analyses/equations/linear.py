from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def linear_equation(first_constant, second_constant, precision = 4):
    """
    Generates a linear function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the linear term of the resultant linear function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the resultant linear function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    evaluation : func
        Function for evaluating a linear equation when passed any integer or float argument

    See Also
    --------
    :func:`~regressions.analyses.derivatives.linear.linear_derivatives`, :func:`~regressions.analyses.integrals.linear.linear_integral`, :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.models.linear.linear_model`

    Notes
    -----
    - Standard form of a linear function: :math:`f(x) = a\\cdot{x} + b`
    - |linear_functions|

    Examples
    --------
    Import `linear_equation` function from `regressions` library
        >>> from regressions.analyses.equations.linear import linear_equation
    Create a linear function with coefficients 2 and 3, then evaluate it at 10
        >>> evaluation_first = linear_equation(2, 3)
        >>> print(evaluation_first(10))
        23.0
    Create a linear function with coefficients -2 and 3, then evaluate it at 10
        >>> evaluation_second = linear_equation(-2, 3)
        >>> print(evaluation_second(10))
        -17.0
    Create a linear function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = linear_equation(0, 0)
        >>> print(evaluation_zero(10))
        0.0011
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create evaluation
    def linear_evaluation(variable):
        evaluation = coefficients[0] * variable + coefficients[1]
        result = rounded_value(evaluation, precision)
        return result
    return linear_evaluation