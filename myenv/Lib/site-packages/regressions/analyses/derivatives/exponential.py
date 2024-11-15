from math import log
from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value, rounded_list

def exponential_derivatives(first_constant, second_constant, precision = 4):
    """
    Calculates the first and second derivatives of an exponential function

    Parameters
    ----------
    first_constant : int or float
        Constant multiple of the original exponential function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Base rate of variable of the original exponential function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    :func:`~regressions.analyses.equations.exponential.exponential_equation`, :func:`~regressions.analyses.integrals.exponential.exponential_integral`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.models.exponential.exponential_model`

    Notes
    -----
    - Standard form of an exponential function: :math:`f(x) = a\\cdot{b^x}`
    - First derivative of an exponential function: :math:`f'(x) = a\\cdot{\\ln{b}\\cdot{b^x}}`
    - Second derivative of an exponential function: :math:`f''(x) = a\\cdot{\\ln^2{b}\\cdot{b^x}}`
    - |differentiation_formulas|
    - |exponential|

    Examples
    --------
    Import `exponential_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.exponential import exponential_derivatives
    Generate the derivatives of an exponential function with coefficients 2 and 3, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = exponential_derivatives(2, 3)
        >>> print(derivatives_constants['first']['constants'])
        [2.1972, 3.0]
        >>> print(derivatives_constants['second']['constants'])
        [2.4139, 3.0]
    Generate the derivatives of an exponential function with coefficients -2 and 3, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = exponential_derivatives(-2, 3)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        -129742.4628
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        -142538.3811
    Generate the derivatives of an exponential function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = exponential_derivatives(0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [-0.0009, 0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [0.0083, 0.0001]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create first derivative
    first_coefficients = [coefficients[0] * log(abs(coefficients[1])), coefficients[1]]
    first_constants = rounded_list(first_coefficients, precision)
    def first_derivative(variable):
        evaluation = first_constants[0] * first_constants[1]**variable
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_coefficients = [first_constants[0] * log(abs(first_constants[1])), first_constants[1]]
    second_constants = rounded_list(second_coefficients, precision)
    def second_derivative(variable):
        evaluation = second_constants[0] * second_constants[1]**variable
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