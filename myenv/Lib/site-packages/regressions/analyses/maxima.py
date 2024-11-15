from regressions.errors.analyses import select_equations
from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import positive_integer
from .intervals import sign_chart

def maxima_points(equation_type, coefficients, precision = 4):
    """
    Calculates the maxima of a specific function

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which the maxima must be determined (e.g., 'linear', 'quadratic')
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
    points : list of float
        Values of the x-coordinates at which the original function has a relative maximum; if the function is sinusoidal, then only two or three results within a two-period interval will be listed; if the function has no maxima, then it will return a list of `None`

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Graphical analysis: :func:`~regressions.analyses.criticals.critical_points`, :func:`~regressions.analyses.intervals.sign_chart`, :func:`~regressions.analyses.minima.minima_points`, :func:`~regressions.analyses.extrema.extrema_points`, :func:`~regressions.analyses.points.key_coordinates`

    Notes
    -----
    - Critical points for the derivative of a function: :math:`c_i = \\{ c_1, c_2, c_3,  \\cdots, c_{n-1}, c_n \\}`
    - X-coordinates of the maxima of the function: :math:`x_{max} = \\{ x \\mid x \\in c_i, f'(\\frac{c_{j-1} + c_j}{2}) > 0, f'(\\frac{c_j + c_{j+1}}{2}) < 0 \\}`
    - |maxima_values|

    Examples
    --------
    Import `maxima_points` function from `regressions` library
        >>> from regressions.analyses.maxima import maxima_points
    Calculate the maxima of a cubic function with coefficients 1, -15, 63, and -7
        >>> points_cubic = maxima_points('cubic', [1, -15, 63, -7])
        >>> print(points_cubic)
        [3.0]
    Calculate the maxima of a sinusoidal function with coefficients 2, 3, 5, and 7
        >>> points_sinusoidal = maxima_points('sinusoidal', [2, 3, 5, 7])
        >>> print(points_sinusoidal)
        [5.5236, 7.618, 9.7124]
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    positive_integer(precision)
    
    # Create sign chart
    intervals = sign_chart(equation_type, coefficients, 1, precision)
    
    # Determine maxima
    result = []
    for i in range(len(intervals)):
        try:
            if intervals[i] == 'positive' and intervals[i + 2] == 'negative':
                result.append(intervals[i + 1])
        except IndexError:
            pass
    
    # Handle no maxima
    if len(result) == 0:
        result.append(None)
    return result