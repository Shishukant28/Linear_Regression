from regressions.errors.scalars import select_integers, positive_integer
from regressions.errors.vectors import vector_of_scalars
from regressions.errors.analyses import select_equations
from .roots.linear import linear_roots_first_derivative, linear_roots_second_derivative
from .roots.quadratic import quadratic_roots_first_derivative, quadratic_roots_second_derivative
from .roots.cubic import cubic_roots_first_derivative, cubic_roots_second_derivative
from .roots.hyperbolic import hyperbolic_roots_first_derivative, hyperbolic_roots_second_derivative
from .roots.exponential import exponential_roots_first_derivative, exponential_roots_second_derivative
from .roots.logarithmic import logarithmic_roots_first_derivative, logarithmic_roots_second_derivative
from .roots.logistic import logistic_roots_first_derivative, logistic_roots_second_derivative
from .roots.sinusoidal import sinusoidal_roots_first_derivative, sinusoidal_roots_second_derivative

def critical_points(equation_type, coefficients, derivative_level, precision = 4):
    """
    Calculates the critical points of a specific function at a certain derivative level

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which critical points must be determined (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients to use to generate the equation to investigate
    derivative_level : int
        Integer corresponding to which derivative to investigate for critical points (1 for the first derivative and 2 for the second derivative)
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the results

    Raises
    ------
    ValueError
        First argument must be either 'linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', or 'sinusoidal'
    TypeError
        Second argument must be a 1-dimensional list containing elements that are integers or floats
    ValueError
        Third argument must be one of the following integers: [1, 2]
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    points : list of float or str
        Values of the x-coordinates at which the original function's derivative either crosses the x-axis or does not exist; if the function is sinusoidal, then only five results within a two-period interval will be listed, but a general form will also be included; if the derivative has no critical points, then it will return a list of `None`

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Graphical analysis: :func:`~regressions.analyses.intervals.sign_chart`, :func:`~regressions.analyses.points.key_coordinates`

    Notes
    -----
    - Domain of a function: :math:`x_i = \\{ x_1, x_2, \\cdots, x_n \\}`
    - Potential critical points of the derivative of the function: :math:`x_c = \\{ c \\mid c \\in x_i, f'(c) = 0 \\cup f'(c) = \\varnothing \\}`
    - |critical_points|

    Examples
    --------
    Import `critical_points` function from `regressions` library
        >>> from regressions.analyses.criticals import critical_points
    Calulate the critical points of the second derivative of a cubic function with coefficients 2, 3, 5, and 7
        >>> points_cubic = critical_points('cubic', [2, 3, 5, 7], 2)
        >>> print(points_cubic)
        [-0.5]
    Calulate the critical points of the first derivative of a sinusoidal function with coefficients 2, 3, 5, and 7
        >>> points_sinusoidal = critical_points('sinusoidal', [2, 3, 5, 7], 1)
        >>> print(points_sinusoidal)
        [5.5236, 6.5708, 7.618, 8.6652, 9.7124, '5.5236 + 1.0472k']
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    select_integers(derivative_level, [1, 2], 'third')
    positive_integer(precision)

    # Create list to return
    results = []

    # Determine critical points for first derivative based on equation type
    if derivative_level == 1:
        if equation_type == 'linear':
            results = linear_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'quadratic':
            results = quadratic_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'cubic':
            results = cubic_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'hyperbolic':
            results = hyperbolic_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'exponential':
            results = exponential_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'logarithmic':
            results = logarithmic_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'logistic':
            results = logistic_roots_first_derivative(*coefficients, precision)
        elif equation_type == 'sinusoidal':
            results = sinusoidal_roots_first_derivative(*coefficients, precision)

    # Determine critical points for second derivative based on equation type
    elif derivative_level == 2:
        if equation_type == 'linear':
            results = linear_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'quadratic':
            results = quadratic_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'cubic':
            results = cubic_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'hyperbolic':
            results = hyperbolic_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'exponential':
            results = exponential_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'logarithmic':
            results = logarithmic_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'logistic':
            results = logistic_roots_second_derivative(*coefficients, precision)
        elif equation_type == 'sinusoidal':
            results = sinusoidal_roots_second_derivative(*coefficients, precision)
    return results