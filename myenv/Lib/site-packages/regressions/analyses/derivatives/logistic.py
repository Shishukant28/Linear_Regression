from math import exp
from regressions.errors.scalars import three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def logistic_derivatives(first_constant, second_constant, third_constant, precision = 4):
    """
    Calculates the first and second derivatives of a logistic function

    Parameters
    ----------
    first_constant : int or float
        Carrying capacity of the original logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Growth rate of the original logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Value of the sigmoid's midpoint of the original logistic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    derivatives['first']['constants'] : list of float
        Coefficients of the resultant first derivative
    derivatives['first']['evaluation'] : func
        Function for evaluating the resultant first derivative at any float or integer argument
    derivatives['second']['constants'] : list of float
        Coefficients of the resultant second derivative
    derivatives['second']['evaluation'] : func
        Function for evaluating the resultant second derivative at any float or integer argument

    See Also
    --------
    :func:`~regressions.analyses.equations.logistic.logistic_equation`, :func:`~regressions.analyses.integrals.logistic.logistic_integral`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.models.logistic.logistic_model`

    Notes
    -----
    - Standard form of a logistic function: :math:`f(x) = \\frac{a}{1 + \\text{e}^{-b\\cdot(x - c)}}`
    - First derivative of a logistic function: :math:`f'(x) = \\frac{ab\\cdot{\\text{e}^{-b\\cdot(x - c)}}}{(1 + \\text{e}^{-b\\cdot(x - c)})^2}`
    - Second derivative of a logistic function: :math:`f''(x) = \\frac{2ab^2\\cdot{\\text{e}^{-2b\\cdot(x - c)}}}{(1 + \\text{e}^{-b\\cdot(x - c)})^3} - \\frac{ab^2\\cdot{\\text{e}^{-b\\cdot(x - c)}}}{(1 + \\text{e}^{-b\\cdot(x - c)})^2}`
    - |differentiation_formulas|
    - |chain_rule|
    - |exponential|

    Examples
    --------
    Import `logistic_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.logistic import logistic_derivatives
    Generate the derivatives of a logistic function with coefficients 2, 3, and 5, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = logistic_derivatives(2, 3, 5)
        >>> print(derivatives_constants['first']['constants'])
        [6.0, 3.0, 5.0]
        >>> print(derivatives_constants['second']['constants'])
        [18.0, 3.0, 5.0]
    Generate the derivatives of a logistic function with coefficients 100, 5, and 11, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = logistic_derivatives(100, 5, 11)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        3.324
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        16.3977
    Generate the derivatives of a logistic function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = logistic_derivatives(0, 0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [0.0001, 0.0001, 0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [0.0001, 0.0001, 0.0001]
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create first derivative
    first_coefficients = [coefficients[0] * coefficients[1], coefficients[1], coefficients[2]]
    first_constants = rounded_list(first_coefficients, precision)
    def first_derivative(variable):
        exponential = exp(-1 * first_constants[1] * (variable - first_constants[2]))
        evaluation = first_constants[0] * exponential * (1 + exponential)**(-2)
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_coefficients = [first_constants[0] * first_constants[1], first_constants[1], first_constants[2]]
    second_constants = rounded_list(second_coefficients, precision)
    def second_derivative(variable):
        exponential = exp(-1 * second_constants[1] * (variable - second_constants[2]))
        evaluation = second_constants[0] * exponential * (1 + exponential)**(-2) * (2 * exponential / (1 + exponential) - 1)
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    second_dictionary = {
        'constants': second_constants,
        'evaluation': second_derivative
    }
    
    # Package both derivatives in single dictionary
    results = {
        'first': first_dictionary,
        'second': second_dictionary
    }
    return results