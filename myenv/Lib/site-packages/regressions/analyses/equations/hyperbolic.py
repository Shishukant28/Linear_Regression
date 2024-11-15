from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def hyperbolic_equation(first_constant, second_constant, precision = 4):
    """
    Generates a hyperbolic function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the reciprocal variable of the resultant hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the resultant hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating a hyperbolic equation when passed any integer or float argument; if zero inputted as argument, it will be converted to a small, non-zero decimal value (e.g., 0.0001)

    See Also
    --------
    :func:`~regressions.analyses.derivatives.hyperbolic.hyperbolic_derivatives`, :func:`~regressions.analyses.integrals.hyperbolic.hyperbolic_integral`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.models.hyperbolic.hyperbolic_model`

    Notes
    -----
    - Standard form of a hyperbolic function: :math:`f(x) = a\\cdot{\\frac{1}{x}} + b`
    - |rational_functions|

    Examples
    --------
    Import `hyperbolic_equation` function from `regressions` library
        >>> from regressions.analyses.equations.hyperbolic import hyperbolic_equation
    Create a hyperbolic function with coefficients 2 and 3, then evaluate it at 10
        >>> evaluation_first = hyperbolic_equation(2, 3)
        >>> print(evaluation_first(10))
        3.2
    Create a hyperbolic function with coefficients -2 and 3, then evaluate it at 10
        >>> evaluation_second = hyperbolic_equation(-2, 3)
        >>> print(evaluation_second(10))
        2.8
    Create a hyperbolic function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = hyperbolic_equation(0, 0)
        >>> print(evaluation_zero(10))
        0.0001
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create evaluation
    def hyperbolic_evaluation(variable):
        # Circumvent division by zero
        if variable == 0:
            variable = 10**(-precision)
        evaluation = coefficients[0] / variable + coefficients[1]
        result = rounded_value(evaluation, precision)
        return result
    return hyperbolic_evaluation