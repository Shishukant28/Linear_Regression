from regressions.errors.scalars import three_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def quadratic_derivatives(first_constant, second_constant, third_constant, precision = 4):
    """
    Calculates the first and second derivatives of a quadratic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the quadratic term of the original quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the linear term of the original quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    third_constant : int or float
        Coefficient of the constant term of the original quadratic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    :func:`~regressions.analyses.equations.quadratic.quadratic_equation`, :func:`~regressions.analyses.integrals.quadratic.quadratic_integral`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.models.quadratic.quadratic_model`

    Notes
    -----
    - Standard form of a quadratic function: :math:`f(x) = a\\cdot{x^2} + b\\cdot{x} + c`
    - First derivative of a quadratic function: :math:`f'(x) = 2a\\cdot{x} + b`
    - Second derivative of a quadratic function: :math:`f''(x) = 2a`
    - |differentiation_formulas|

    Examples
    --------
    Import `quadratic_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.quadratic import quadratic_derivatives
    Generate the derivatives of a quadratic function with coefficients 2, 3, and 5, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = quadratic_derivatives(2, 3, 5)
        >>> print(derivatives_constants['first']['constants'])
        [4.0, 3.0]
        >>> print(derivatives_constants['second']['constants'])
        [4.0]
    Generate the derivatives of a quadratic function with coefficients 7, -5, and 3, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = quadratic_derivatives(7, -5, 3)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        135.0
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        14.0
    Generate the derivatives of a quadratic function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = quadratic_derivatives(0, 0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [0.0002, 0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [0.0002]
    """
    # Handle input errors
    three_scalars(first_constant, second_constant, third_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant, third_constant], precision)

    # Create first derivative
    first_constants = [2 * coefficients[0], coefficients[1]]
    def first_derivative(variable):
        evaluation = first_constants[0] * variable + first_constants[1]
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_constants = [first_constants[0]]
    def second_derivative(variable):
        evaluation = second_constants[0]
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