from regressions.errors.scalars import two_scalars, positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.statistics.rounding import rounded_value

def hyperbolic_derivatives(first_constant, second_constant, precision = 4):
    """
    Calculates the first and second derivatives of a hyperbolic function

    Parameters
    ----------
    first_constant : int or float
        Coefficient of the reciprocal variable of the original hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    second_constant : int or float
        Coefficient of the constant term of the original hyperbolic function; if zero, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
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
        Function for evaluating the resultant first derivative at any float or integer argument; if zero inputted as argument, it will be converted to a small, non-zero decimal value (e.g., 0.0001)
    derivatives['second']['constants'] : list of float
        Coefficients of the resultant second derivative
    derivatives['second']['evaluation'] : func
        Function for evaluating the resultant second derivative at any float or integer argument; if zero inputted as argument, it will be converted to a small, non-zero decimal value (e.g., 0.0001)

    See Also
    --------
    :func:`~regressions.analyses.equations.hyperbolic.hyperbolic_equation`, :func:`~regressions.analyses.integrals.hyperbolic.hyperbolic_integral`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.models.hyperbolic.hyperbolic_model`

    Notes
    -----
    - Standard form of a hyperbolic function: :math:`f(x) = a\\cdot{\\frac{1}{x}} + b`
    - First derivative of a hyperbolic function: :math:`f'(x) = -a\\cdot{\\frac{1}{x^2}}`
    - Second derivative of a hyperbolic function: :math:`f''(x) = 2a\\cdot{\\frac{1}{x^3}}`
    - |differentiation_formulas|

    Examples
    --------
    Import `hyperbolic_derivatives` function from `regressions` library
        >>> from regressions.analyses.derivatives.hyperbolic import hyperbolic_derivatives
    Generate the derivatives of a hyperbolic function with coefficients 2 and 3, then display the coefficients of its first and second derivatives
        >>> derivatives_constants = hyperbolic_derivatives(2, 3)
        >>> print(derivatives_constants['first']['constants'])
        [-2.0]
        >>> print(derivatives_constants['second']['constants'])
        [4.0]
    Generate the derivatives of a hyperbolic function with coefficients -2 and 3, then evaluate its first and second derivatives at 10
        >>> derivatives_evaluation = hyperbolic_derivatives(-2, 3)
        >>> print(derivatives_evaluation['first']['evaluation'](10))
        0.02
        >>> print(derivatives_evaluation['second']['evaluation'](10))
        -0.004
    Generate the derivatives of a hyperbolic function with all inputs set to 0, then display the coefficients of its first and second derivatives
        >>> derivatives_zeroes = hyperbolic_derivatives(0, 0)
        >>> print(derivatives_zeroes['first']['constants'])
        [-0.0001]
        >>> print(derivatives_zeroes['second']['constants'])
        [0.0002]
    """
    # Handle input errors
    two_scalars(first_constant, second_constant)
    positive_integer(precision)
    coefficients = no_zeroes([first_constant, second_constant], precision)

    # Create first derivative
    first_constants = [-1 * coefficients[0]]
    def first_derivative(variable):
        # Circumvent division by zero
        if variable == 0:
            variable = 10**(-precision)
        evaluation = first_constants[0] / variable**2
        rounded_evaluation = rounded_value(evaluation, precision)
        return rounded_evaluation
    first_dictionary = {
        'constants': first_constants,
        'evaluation': first_derivative
    }

    # Create second derivative
    second_constants = [-2 * first_constants[0]]
    def second_derivative(variable):
        # Circumvent division by zero
        if variable == 0:
            variable = 10**(-precision)
        evaluation = second_constants[0] / variable**3
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