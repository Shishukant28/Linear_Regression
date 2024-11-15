from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def exponential_equation(first_constant, second_constant, precision = 4):
    """
    Generates an exponential function to provide evaluations at variable inputs

    Parameters
    ----------
    first_constant : int or float
        Constant multiple of the resultant exponential function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Base rate of variable of the resultant exponential function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating an exponential equation when passed any integer or float argument
    
    See Also
    --------
    :func:`~regressions.analyses.derivatives.exponential.exponential_derivatives`, :func:`~regressions.analyses.integrals.exponential.exponential_integral`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.models.exponential.exponential_model`

    Notes
    -----
    - Standard form of an exponential function: :math:`f(x) = a\\cdot{b^x}`
    - |exponential_functions|

    Examples
    --------
    Import `exponential_equation` function from `regressions` library
        >>> from regressions.analyses.equations.exponential import exponential_equation
    Create an exponential function with coefficients 2 and 3, then evaluate it at 10
        >>> evaluation_first = exponential_equation(2, 3)
        >>> print(evaluation_first(10))
        118098.0
    Create an exponential function with coefficients -2 and 3, then evaluate it at 10
        >>> evaluation_second = exponential_equation(-2, 3)
        >>> print(evaluation_second(10))
        -118098.0
    Create an exponential function with all inputs set to 0, then evaluate it at 10
        >>> evaluation_zero = exponential_equation(0, 0)
        >>> print(evaluation_zero(10))
        0.0001
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create evaluation
    def exponential_evaluation(variable):
        evaluation = coefficients[0] * coefficients[1]**variable
        result = rounded_value(evaluation, precision)
        return result
    return exponential_evaluation