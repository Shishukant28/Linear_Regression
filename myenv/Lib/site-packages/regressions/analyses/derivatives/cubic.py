from regressions.errors.scalars import four_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def cubic_derivatives(first_constant, second_constant, third_constant, fourth_constant, precision = 4):
    """
    Calculates the first and second derivatives of a cubic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the cubic term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the quadratic term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Coefficient of the linear term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    fourth_constant : int or float
        Coefficient of the constant term of the original cubic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    :func:`~regressions.analyses.equations.cubic.cubic_equation`, :func:`~regressions.analyses.integrals.cubic.cubic_integral`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.models.cubic.cubic_model`

    Notes
    -----
    - Standard form of a cubic function: :math:`f(x) = a\\cdot{x^3} + b\\cdot{x^2} + c\\cdot{x} + d`
    - First derivative of a cubic function: :math:`f'(x) = 3a\\cdot{x^2} + 2b\\cdot{x} + c`
    - Second derivative of a cubic function: :math:`f''(x) = 6a\\cdot{x} + 2b`
    - |differentiation_formulas|

    Examples
    --------
    Import `cubic_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.cubic import cubic_derivatives
    Generate the derivatives of a cubic function with coefficients 2, 3, 5, and 7, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = cubic_derivatives(2, 3, 5, 7)
        >>> print(derivatives_constants['first']['constants'])
        [6.0, 6.0, 5.0]
        >>> print(derivatives_constants['second']['constants'])
        [12.0, 6.0]
    Generate the derivatives of a cubic function with coefficients 7, -5, -3, and 2, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = cubic_derivatives(7, -5, -3, 2)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        1997.0
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        410.0
    Generate the derivatives of a cubic function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = cubic_derivatives(0, 0, 0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [0.0003, 0.0002, 0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [0.0006, 0.0002]
    """
    # Handle input errors
    four_scalars(first_constant, second_constant, third_constant, fourth_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant, fourth_constant], precision)

    # Create first derivative
    first_constants = [3 * coefficients[0], 2 * coefficients[1], coefficients[2]]
    def first_derivative(variable):
        evaluation = first_constants[0] * variable**2 + first_constants[1] * variable + first_constants[2]
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_constants = [2 * first_constants[0], first_constants[1]]
    def second_derivative(variable):
        evaluation = second_constants[0] * variable + second_constants[1]
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