from math import sin
from regressions.errors.scalars import four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def sinusoidal_equation(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Generates a sinusoidal function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Vertical stretch factor of the resultant sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Horizontal stretch factor of the resultant sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Horizontal shift of the resultant sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    fourth_constant : int or float
        Vertical shift of the resultant sine function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating a sinusoidal equation when passed any integer or float argument

    See Also
    --------
    :func:`~regressions.analyses.derivatives.sinusoidal.sinusoidal_derivatives`, :func:`~regressions.analyses.integrals.sinusoidal.sinusoidal_integral`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`, :func:`~regressions.models.sinusoidal.sinusoidal_model`

    Notes
    -----
    - Standard form of a sinusoidal function: :math:`f(x) = a\\cdot{\\sin(b\\cdot(x - c))} + d`

        - Period of function: :math:`\\frac{2\\pi}{|b|}`
        - Amplitude of function: :math:`|a|`

    - |sine_functions|

    Examples
    --------
    Import `sinusoidal_equation` function from `regressions` library
        >>> from regressions.analyses.equations.sinusoidal import sinusoidal_equation
    Create a sinusoidal function with coefficients 2, 3, 5, and 7, then evaluate it at 10
        >>> evaluation_first = sinusoidal_equation(2, 3, 5, 7)
        >>> print(evaluation_first(10))
        8.3006
    Create a sinusoidal function with coefficients 7, -5, -3, and 2, then evaluate it at 10
        >>> evaluation_second = sinusoidal_equation(7, -5, -3, 2)
        >>> print(evaluation_second(10))
        -3.7878
    Create a sinusoidal function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = sinusoidal_equation(0, 0, 0, 0)
        >>> print(evaluation_zero(10))
        0.0001
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Create evaluation
    def sinusoidal_evaluation(variable):
        evaluation = coefficients[0] * sin(coefficients[1] * (variable - coefficients[2])) + coefficients[3]
        result = rounded_value(evaluation, precision)
        return result
    return sinusoidal_evaluation