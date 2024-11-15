from regressions.errors.analyses import select_equations
from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import positive_integer
from .intervals import sign_chart

def inflection_points(equation_type, coefficients, precision = 4):
    """
    Calculates the inflection points of a specific function

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which inflections must be determined (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients to use to generate the equation to investigate
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the results

    Raises
    ------
    ValueError
        First argument must be either 'linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', or 'sinusoidal'
    TypeError
        Second argument must be a 1-dimensional list containing elements that are integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    points : list of float or str
        Values of the x-coordinates at which the original function has an inflection point; if the function is sinusoidal, then only five results within a two-period interval will be listed, but a general form will also be included; if the function has no inflection points, then it will return a list of `None`

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Graphical analysis: :func:`~regressions.analyses.criticals.critical_points`, :func:`~regressions.analyses.intervals.sign_chart`, :func:`~regressions.analyses.points.key_coordinates`

    Notes
    -----
    - Critical points for the second derivative of a function: :math:`c_i = \\{ c_1, c_2, c_3,  \\cdots, c_{n-1}, c_n \\}`
    - X-coordinates of the inflections of the function: :math:`x_{infl} = \\{ x \\mid x \\in c_i, \\left( f''(\\frac{c_{j-1} + c_j}{2}) < 0 \\cap f''(\\frac{c_j + c_{j+1}}{2}) > 0 \\right) \\\\ \\cup \\left( f''(\\frac{c_{j-1} + c_j}{2}) > 0 \\cap f''(\\frac{c_j + c_{j+1}}{2}) < 0 \\right) \\}`
    - |inflection_points|

    Examples
    --------
    Import `inflection_points` function from `regressions` library
        >>> from regressions.analyses.inflections import inflection_points
    Calculate the inflection points of a cubic functions with coefficients 1, -15, 63, and -7
        >>> points_cubic = inflection_points('cubic', [1, -15, 63, -7])
        >>> print(points_cubic)
        [5.0]
    Calculate the inflection points of a sinusoidal functions with coefficients 2, 3, 5, and 7
        >>> points_sinusoidal = inflection_points('sinusoidal', [2, 3, 5, 7])
        >>> print(points_sinusoidal)
        [5.0, 6.0472, 7.0944, 8.1416, 9.1888, '5.0 + 1.0472k']
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    positive_integer(precision)

    # Create sign chart
    intervals_set = sign_chart(equation_type, coefficients, 2, precision)

    # Handle hyperbolic case
    if equation_type == 'hyperbolic':
        result = [None]
        return result
    
    # Determine inflections
    result = []
    for i in range(len(intervals_set)):
        try:
            if (intervals_set[i] == 'positive' and intervals_set[i + 2] == 'negative') or (intervals_set[i] == 'negative' and intervals_set[i + 2] == 'positive'):
                result.append(intervals_set[i + 1])
        except IndexError:
            pass
    
    # Handle sinusoidal case
    if equation_type == 'sinusoidal':
        result.append(intervals_set[-1])

    # Handle no inflections
    if len(result) == 0:
        result.append(None)
    return result