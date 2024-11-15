from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes

def linear_derivatives(first_constant, second_constant, precision = 4):
    """
    Calculates the first and second derivatives of a linear function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the linear term of the original linear function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the original linear function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
    :func:`~regressions.analyses.equations.linear.linear_equation`, :func:`~regressions.analyses.integrals.linear.linear_integral`, :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.models.linear.linear_model`

    Notes
    -----
    - Standard form of a linear function: :math:`f(x) = a\\cdot{x} + b`
    - First derivative of a linear function: :math:`f'(x) = a`
    - Second derivative of a linear function: :math:`f''(x) = 0`
    - |differentiation_formulas|

    Examples
    --------
    Import `linear_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.linear import linear_derivatives
    Generate the derivatives of a linear function with coefficients 2 and 3, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = linear_derivatives(2, 3)
        >>> print(derivatives_constants['first']['constants'])
        [2.0]
        >>> print(derivatives_constants['second']['constants'])
        [0.0]
    Generate the derivatives of a linear function with coefficients -2 and 3, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = linear_derivatives(-2, 3)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        -2.0
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        0.0
    Generate the derivatives of a linear function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = linear_derivatives(0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [0.0]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Creat first derivative
    first_constants = [coefficients[0]]
    def first_derivative(variable):
        evaluation = first_constants[0]
        return evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_constants = [0.0]
    def second_derivative(variable):
        evaluation = second_constants[0]
        return evaluation
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