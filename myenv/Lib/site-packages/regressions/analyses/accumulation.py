from regressions.errors.scalars import compare_scalars, positive_integer
from regressions.errors.vectors import vector_of_scalars
from regressions.errors.analyses import select_equations
from regressions.statistics.rounding import rounded_value
from .integrals.linear import linear_integral
from .integrals.quadratic import quadratic_integral
from .integrals.cubic import cubic_integral
from .integrals.hyperbolic import hyperbolic_integral
from .integrals.exponential import exponential_integral
from .integrals.logarithmic import logarithmic_integral
from .integrals.logistic import logistic_integral
from .integrals.sinusoidal import sinusoidal_integral

def accumulated_area(equation_type, coefficients, start, end, precision = 4):
    """
    Evaluates the definite integral between two points for a specific function

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which the definite integral must be evaluated (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients of the original function to integrate
    start : int or float
        Value of the x-coordinate of the first point to use for evaluating the definite integral
    end : int or float
        Value of the x-coordinate of the second point to use for evaluating the definite integral
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the result

    Raises
    ------
    ValueError
        First argument must be either 'linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', or 'sinusoidal'
    TypeError
        Second argument must be a 1-dimensional list containing elements that are integers or floats
    TypeError
        Third and fourth arguments must be integers or floats
    ValueError
        Third argument must be less than or equal to fourth argument
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    area : float
        Definite integral of the original equation, evaluated between two points; if start and end values are identical, then definite integral will be zero

    See Also
    --------
    :func:`~regressions.analyses.integrals.linear.linear_integral`, :func:`~regressions.analyses.integrals.quadratic.quadratic_integral`, :func:`~regressions.analyses.integrals.cubic.cubic_integral`, :func:`~regressions.analyses.integrals.hyperbolic.hyperbolic_integral`, :func:`~regressions.analyses.integrals.exponential.exponential_integral`, :func:`~regressions.analyses.integrals.logarithmic.logarithmic_integral`, :func:`~regressions.analyses.integrals.logistic.logistic_integral`, :func:`~regressions.analyses.integrals.sinusoidal.sinusoidal_integral`

    Notes
    -----
    - Definite integral of a function: :math:`\\int_{a}^{b} f(x) \\,dx = F(b) - F(a)`
    - |definite_integral|
    - |fundamental_theorem|

    Examples
    --------
    Import `accumulated_area` function from `regressions` library
        >>> from regressions.analyses.accumulation import accumulated_area
    Evaluate the definite integral of a linear function with coefficients 2 and 3 between the end points 10 and 20
        >>> area_linear = accumulated_area('linear', [2, 3], 10, 20)
        >>> print(area_linear)
        330.0
    Evaluate the definite integral of a cubic function with coefficients 8, 6, -10, and 7 between the end points 10 and 20
        >>> area_cubic = accumulated_area('cubic', [8, 6, -10, 7], 10, 20)
        >>> print(area_cubic)
        312570.0
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    compare_scalars(start, end, 'third', 'fourth')
    positive_integer(precision)

    # Create indefinite integral based on equation type
    integral = lambda x : x
    if equation_type == 'linear':
        integral = linear_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'quadratic':
        integral = quadratic_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'cubic':
        integral = cubic_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'hyperbolic':
        integral = hyperbolic_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'exponential':
        integral = exponential_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'logarithmic':
        integral = logarithmic_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'logistic':
        integral = logistic_integral(*coefficients, precision)['evaluation']
    elif equation_type == 'sinusoidal':
        integral = sinusoidal_integral(*coefficients, precision)['evaluation']
    
    # Evaluate definite integral
    area = integral(end) - integral(start)

    # Round evaluation
    result = rounded_value(area, precision)
    return result