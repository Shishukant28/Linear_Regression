from math import log
from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def logarithmic_equation(first_constant, second_constant, precision = 4):
    """
    Generates a logarithmic function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the logarithmic term of the resultant logarithmic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the resultant logarithmic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating a logarithmic equation when passed any integer or float argument; if zero inputted as argument, it will be converted to a small, non-zero decimal value (e.g., 0.0001)

    See Also
    --------
    :func:`~regressions.analyses.derivatives.logarithmic.logarithmic_derivatives`, :func:`~regressions.analyses.integrals.logarithmic.logarithmic_integral`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.models.logarithmic.logarithmic_model`

    Notes
    -----
    - Standard form of a logarithmic function: :math:`f(x) = a\\cdot{\\ln{x}} + b`
    - |logarithmic_functions|

    Examples
    --------
    Import `logarithmic_equation` function from `regressions` library
        >>> from regressions.analyses.equations.logarithmic import logarithmic_equation
    Create a logarithmic function with coefficients 2 and 3, then evaluate it at 10
        >>> evaluation_first = logarithmic_equation(2, 3)
        >>> print(evaluation_first(10))
        7.6052
    Create a logarithmic function with coefficients -2 and 3, then evaluate it at 10
        >>> evaluation_second = logarithmic_equation(-2, 3)
        >>> print(evaluation_second(10))
        -1.6052
    Create a logarithmic function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = logarithmic_equation(0, 0)
        >>> print(evaluation_zero(10))
        0.0003
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create evaluation
    def logarithmic_evaluation(variable):
        # Circumvent logarithm of zero
        if variable == 0:
            variable = 10**(-precision)
        evaluation = coefficients[0] * log(abs(variable)) + coefficients[1]
        result = rounded_value(evaluation, precision)
        return result
    return logarithmic_evaluation