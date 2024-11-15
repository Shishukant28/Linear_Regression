from regressions.errors.scalars import three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def quadratic_equation(first_constant, second_constant, third_constant, precision = 4):
    """
    Generates a quadratic function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the quadratic term of the resultant quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the linear term of the resultant quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Coefficient of the constant term of the resultant quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating a quadratic equation when passed any integer or float argument

    See Also
    --------
    :func:`~regressions.analyses.derivatives.quadratic.quadratic_derivatives`, :func:`~regressions.analyses.integrals.quadratic.quadratic_integral`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.models.quadratic.quadratic_model`

    Notes
    -----
    - Standard form of a quadratic function: :math:`f(x) = a\\cdot{x^2} + b\\cdot{x} + c`
    - |quadratic_functions|

    Examples
    --------
    Import `quadratic_equation` function from `regressions` library
        >>> from regressions.analyses.equations.quadratic import quadratic_equation
    Create a quadratic function with coefficients 2, 3, and 5, then evaluate it at 10
        >>> evaluation_first = quadratic_equation(2, 3, 5)
        >>> print(evaluation_first(10))
        235.0
    Create a quadratic function with coefficients 7, -5, and 3, then evaluate it at 10
        >>> evaluation_second = quadratic_equation(7, -5, 3)
        >>> print(evaluation_second(10))
        653.0
    Create a quadratic function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = quadratic_equation(0, 0, 0)
        >>> print(evaluation_zero(10))
        0.0111
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create evaluation
    def quadratic_evaluation(variable):
        evaluation = coefficients[0] * variable**2 + coefficients[1] * variable + coefficients[2]
        result = rounded_value(evaluation, precision)
        return result
    return quadratic_evaluation