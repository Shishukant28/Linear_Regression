from regressions.errors.scalars import four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def cubic_equation(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Generates a cubic function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the cubic term of the resultant cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the quadratic term of the resultant cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Coefficient of the linear term of the resultant cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    fourth_constant : int or float
        Coefficient of the constant term of the resultant cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    evaluation : func
        Function for evaluating a cubic equation when passed any integer or float argument
    
    See Also
    --------
    :func:`~regressions.analyses.derivatives.cubic.cubic_derivatives`, :func:`~regressions.analyses.integrals.cubic.cubic_integral`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.models.cubic.cubic_model`

    Notes
    -----
    - Standard form of a cubic function: :math:`f(x) = a\\cdot{x^3} + b\\cdot{x^2} + c\\cdot{x} + d`
    - |cubic_functions|

    Examples
    --------
    Import `cubic_equation` function from `regressions` library
        >>> from regressions.analyses.equations.cubic import cubic_equation
    Create a cubic function with coefficients 2, 3, 5, and 7, then evaluate it at 10
        >>> evaluation_first = cubic_equation(2, 3, 5, 7)
        >>> print(evaluation_first(10))
        2357.0
    Create a cubic function with coefficients 7, -5, -3, and 2, then evaluate it at 10
        >>> evaluation_second = cubic_equation(7, -5, -3, 2)
        >>> print(evaluation_second(10))
        6472.0
    Create a cubic function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = cubic_equation(0, 0, 0, 0)
        >>> print(evaluation_zero(10))
        0.1111
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Create evaluation
    def cubic_evaluation(variable):
        evaluation = coefficients[0] * variable**3 + coefficients[1] * variable**2 + coefficients[2] * variable + coefficients[3]
        result = rounded_value(evaluation, precision)
        return result
    return cubic_evaluation