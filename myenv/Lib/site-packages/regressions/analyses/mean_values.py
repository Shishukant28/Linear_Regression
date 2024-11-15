from regressions.errors.analyses import select_equations
from regressions.errors.scalars import compare_scalars, positive_integer
from regressions.errors.vectors import vector_of_scalars
from regressions.statistics.rounding import rounded_value
from .roots.linear import linear_roots_initial_value, linear_roots_derivative_initial_value
from .roots.quadratic import quadratic_roots_initial_value, quadratic_roots_derivative_initial_value
from .roots.cubic import cubic_roots_initial_value, cubic_roots_derivative_initial_value
from .roots.hyperbolic import hyperbolic_roots_initial_value, hyperbolic_roots_derivative_initial_value
from .roots.exponential import exponential_roots_initial_value, exponential_roots_derivative_initial_value
from .roots.logarithmic import logarithmic_roots_initial_value, logarithmic_roots_derivative_initial_value
from .roots.logistic import logistic_roots_initial_value, logistic_roots_derivative_initial_value
from .roots.sinusoidal import sinusoidal_roots_initial_value, sinusoidal_roots_derivative_initial_value
from .equations.linear import linear_equation
from .equations.quadratic import quadratic_equation
from .equations.cubic import cubic_equation
from .equations.hyperbolic import hyperbolic_equation
from .equations.exponential import exponential_equation
from .equations.logarithmic import logarithmic_equation
from .equations.logistic import logistic_equation
from .equations.sinusoidal import sinusoidal_equation
from .accumulation import accumulated_area
from .points import points_within_range, shifted_points_within_range

def average_value_derivative(equation_type, coefficients, start, end, precision = 4):
    """
    Evaluates the average rate of change between two points for a given function

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which the definite integral must be evaluated (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients of the original function to use for evaluating the average rate of change
    start : int or float
        Value of the x-coordinate of the first point to use for evaluating the rate of change
    end : int or float
        Value of the x-coordinate of the second point to use for evaluating the rate of change
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
    average : float
        Slope of a function between two points; if start and end values are identical, then slope will be zero

    See Also
    --------
    :func:`~regressions.analyses.mean_values.mean_values_derivative`, 
    :func:`~regressions.analyses.mean_values.average_value_integral`, :func:`~regressions.analyses.mean_values.mean_values_integral`

    Notes
    -----
    - Slope of a function over an interval: :math:`m = \\frac{f(b) - f(a)}{b - a}`
    - |mean_derivatives|

    Examples
    --------
    Import `average_value_derivative` function from `regressions` library
        >>> from regressions.analyses.mean_values import average_value_derivative
    Evaluate the average rate of change of a cubic function with coefficients 2, 3, 5, and 7 between end points of 10 and 20
        >>> average_cubic = average_value_derivative('cubic', [2, 3, 5, 7], 10, 20)
        >>> print(average_cubic)
        1495.0
    Evaluate the average rate of change of a sinusoidal function with coefficients 2, 3, 5, and 7 between end points of 10 and 20
        >>> average_sinusoidal = average_value_derivative('sinusoidal', [2, 3, 5, 7], 10, 20)
        >>> print(average_sinusoidal)
        0.0401
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    compare_scalars(start, end, 'third', 'fourth')
    positive_integer(precision)

    # Create equation based on equation type
    equation = lambda x : x
    if equation_type == 'linear':
        equation = linear_equation(*coefficients, precision)
    elif equation_type == 'quadratic':
        equation = quadratic_equation(*coefficients, precision)
    elif equation_type == 'cubic':
        equation = cubic_equation(*coefficients, precision)
    elif equation_type == 'hyperbolic':
        equation = hyperbolic_equation(*coefficients, precision)
    elif equation_type == 'exponential':
        equation = exponential_equation(*coefficients, precision)
    elif equation_type == 'logarithmic':
        equation = logarithmic_equation(*coefficients, precision)
    elif equation_type == 'logistic':
        equation = logistic_equation(*coefficients, precision)
    elif equation_type == 'sinusoidal':
        equation = sinusoidal_equation(*coefficients, precision)
    
    # Create intermediary variables
    vertical_change = equation(end) - equation(start)
    horizontal_change = end - start
    
    # Circumvent division by zero
    if horizontal_change == 0:
        horizontal_change = 10**(-precision)
    
    # Determine average slope
    ratio = vertical_change / horizontal_change
    result = rounded_value(ratio, precision)
    return result

def mean_values_derivative(equation_type, coefficients, start, end, precision = 4):
    """
    Generates a list of all the x-coordinates whose instantaneous rates of change equal the function's average rate of change between two points

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which an average value must be determined (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients to use to generate the equation to investigate
    start : int or float
        Value of the x-coordinate of the first point to use for evaluating the rate of change; all results must be greater than this value
    end : int or float
        Value of the x-coordinate of the second point to use for evaluating the rate of change; all results must be less than this value
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
    points : list of float or str
        Values of the x-coordinates within the specified interval at which the original function has an instantaneous rate of change equal to its average rate of change over that entire interval; if the function is sinusoidal, then only the initial results within at most a four-period interval within the specified interval will be listed, but general forms will also be included (however, their results may be outside the specified interval); if the algorithm cannot determine any values, then it will return a list of `None`

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Mean values: :func:`~regressions.analyses.mean_values.average_value_derivative`, :func:`~regressions.analyses.mean_values.average_value_integral`, :func:`~regressions.analyses.mean_values.mean_values_integral`

    Notes
    -----
    - Mean values for the derivative over an interval: :math:`f'(c) = \\frac{f(b) - f(a)}{b - a}` 
    - |mean_derivatives|

    Examples
    --------
    Import `mean_values_derivative` function from `regressions` library
        >>> from regressions.analyses.mean_values import mean_values_derivative
    Generate a list of all the x-coordinates whose instantaneous rates of change equal the function's average rate of change for a cubic function with coefficients 2, 3, 5, and 7 between end points of 10 and 20
        >>> points_cubic = mean_values_derivative('cubic', [2, 3, 5, 7], 10, 20)
        >>> print(points_cubic)
        [15.2665]
    Generate a list of all the x-coordinates whose instantaneous rates of change equal the function's average rate of change for a cubic function with coefficients 2, 3, 5, and 7 between end points of 10 and 20
        >>> points_sinusoidal = mean_values_derivative('sinusoidal', [2, 3, 5, 7], 10, 20)
        >>> print(points_sinusoidal)
        [10.7618, 11.8046, 12.8562, 13.899, 14.9506, 15.9934, 17.045, 18.0878, 19.1394, '10.7618 + 2.0944k', '11.8046 + 2.0944k']
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    compare_scalars(start, end, 'third', 'fourth')
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine average rate of change over interval
    average = average_value_derivative(equation_type, coefficients, start, end, precision)

    # Circumvent division by zero
    if average == 0:
        average = 10**(-precision)

    # Determine points satisfying theorem based on equation type
    if equation_type == 'linear':
        result = linear_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'quadratic':
        result = quadratic_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'cubic':
        result = cubic_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'hyperbolic':
        result = hyperbolic_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'exponential':
        result = exponential_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'logarithmic':
        result = logarithmic_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'logistic':
        result = logistic_roots_derivative_initial_value(*coefficients, average, precision)
    elif equation_type == 'sinusoidal':
        options = sinusoidal_roots_derivative_initial_value(*coefficients, average, precision)
        result = shifted_points_within_range(options, start, end, precision)
    
    # Eliminate points outside of interval
    final_result = points_within_range(result, start, end)
    return final_result

def average_value_integral(equation_type, coefficients, start, end, precision = 4):
    """
    Evaluates the average value of a given function between two points

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which the definite integral must be evaluated (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients of the original function to integrate
    start : int or float
        Value of the x-coordinate of the first point to use for evaluating the average value
    end : int or float
        Value of the x-coordinate of the second point to use for evaluating the average value
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
    average : float
        Average value of the function between two points; if start and end values are identical, then average value will be zero

    See Also
    --------
    :func:`~regressions.analyses.accumulation.accumulated_area`, :func:`~regressions.analyses.mean_values.mean_values_integral`, :func:`~regressions.analyses.mean_values.average_value_derivative`, :func:`~regressions.analyses.mean_values.mean_values_derivative`

    Notes
    -----
    - Average value of a function over an interval: :math:`f_{avg} = \\frac{1}{b - a}\\cdot{\\int_{a}^{b} f(x) \\,dx}`
    - |mean_integrals|

    Examples
    --------
    Import `average_value_integral` function from `regressions` library
        >>> from regressions.analyses.mean_values import average_value_integral
    Evaluate the average value of a cubic function with coefficients 2, 3, 5, and 7 between end points of 10 and 20
        >>> average_cubic = average_value_integral('cubic', [2, 3, 5, 7], 10, 20)
        >>> print(average_cubic)
        8282.0
    Evaluate the average value of a sinusoidal function with coefficients 2, 3, 5, and 7 between end points of 10 and 20
        >>> average_sinusoidal = average_value_integral('sinusoidal', [2, 3, 5, 7], 10, 20)
        >>> print(average_sinusoidal)
        6.9143
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    compare_scalars(start, end, 'third', 'fourth')
    positive_integer(precision)

    # Determine accumulated value of function over interval
    accumulated_value = accumulated_area(equation_type, coefficients, start, end, precision)

    # Create intermediary variable
    change = end - start

    # Circumvent division by zero
    if change == 0:
        change = 10**(-precision)
    
    # Determine average value of function over interval
    ratio = accumulated_value / change

    # Round value
    rounded_ratio = rounded_value(ratio, precision)

    # Return result
    result = rounded_ratio
    return result

def mean_values_integral(equation_type, coefficients, start, end, precision = 4):
    """
    Generates a list of all the x-coordinates between two points at which a function's value will equal its average value over that interval

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which an average value must be determined (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients of the origianl function under investigation
    start : int or float
        Value of the x-coordinate of the first point to use for evaluating the average value; all results must be greater than this value
    end : int or float
        Value of the x-coordinate of the second point to use for evaluating the average value; all results must be less than this value
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
    points : list of float or str
        Values of the x-coordinates within the specified interval at which the original function has a value equal to its average value over that entire interval; if the function is sinusoidal, then only the initial results within at most a four-period interval within the specified interval will be listed, but general forms will also be included (however, their results may be outside the specified interval); if the algorithm cannot determine any values, then it will return a list of `None`

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Mean values: :func:`~regressions.analyses.mean_values.average_value_integral`, :func:`~regressions.analyses.mean_values.average_value_derivative`, :func:`~regressions.analyses.mean_values.mean_values_derivative`

    Notes
    -----
    - Mean values for the integral over an interval: :math:`f(c) = \\frac{1}{b - a}\\cdot{\\int_{a}^{b} f(x) \\,dx}` 
    - |mean_integrals|

    Examples
    --------
    Import `mean_values_integral` function from `regressions` library
        >>> from regressions.analyses.mean_values import mean_values_integral
    Generate a list of all the x-coordinates of a cubic function with coefficients 2, 3, 5, and 7 at which the function's value will equal its average value between 10 and 20
        >>> points_cubic = mean_values_integral('cubic', [2, 3, 5, 7], 10, 20)
        >>> print(points_cubic)
        [15.5188]
    Generate a list of all the x-coordinates of a sinusoidal function with coefficients 2, 3, 5, and 7 at which the function's value will equal its average value between 10 and 20
        >>> points_sinusoidal = mean_values_integral('sinusoidal', [2, 3, 5, 7], 10, 20)
        >>> print(points_sinusoidal)
        [10.2503, 11.2689, 12.3447, 13.3633, 14.4391, 15.4577, 16.5335, 17.5521, 18.6279, 19.6465, '10.2503 + 2.0944k', '11.2689 + 2.0944k']
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    compare_scalars(start, end, 'third', 'fourth')
    positive_integer(precision)

    # Create list to return
    result = []

    # Determine average value of function over interval
    average = average_value_integral(equation_type, coefficients, start, end, precision)

    # Circumvent division by zero
    if average == 0:
        average = 10**(-precision)
    
    # Determine points satisfying theorem based on equation type
    if equation_type == 'linear':
        result = linear_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'quadratic':
        result = quadratic_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'cubic':
        result = cubic_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'hyperbolic':
        result = hyperbolic_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'exponential':
        result = exponential_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'logarithmic':
        result = logarithmic_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'logistic':
        result = logistic_roots_initial_value(*coefficients, average, precision)
    elif equation_type == 'sinusoidal':
        options = sinusoidal_roots_initial_value(*coefficients, average, precision)
        result = shifted_points_within_range(options, start, end, precision)
    
    # Eliminate points outside of interval
    final_result = points_within_range(result, start, end)
    return final_result

def average_values(equation_type, coefficients, start, end, precision = 4):
    """
    Calculates the average values for a specific function

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which average values must be determined (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients of the origianl function under investigation
    start : int or float
        Value of the x-coordinate of the first point to use for evaluating the average values; results within lists must be greater than this value
    end : int or float
        Value of the x-coordinate of the second point to use for evaluating the average values; results within lists must be less than this value
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
    averages['average_value_derivative'] : float
        Slope of a function between two points
    averages['mean_values_derivative'] : list of float or str
        Values of the x-coordinates within the specified interval at which the original function has a value equal to its average value over that entire interval; if the function is sinusoidal, then only the initial results within at most a four-period interval within the specified interval will be listed, but general forms will also be included (however, their results may be outside the specified interval); if the algorithm cannot determine any values, then it will return a list of `None`
    averages['average_value_integral'] : float
        Average value of the function between two points
    averages['mean_values_integral'] : list of float or str
        Values of the x-coordinates within the specified interval at which the original function has a value equal to its average value over that entire interval; if the function is sinusoidal, then only the initial results within at most a four-period interval within the specified interval will be listed, but general forms will also be included (however, their results may be outside the specified interval); if the algorithm cannot determine any values, then it will return a list of `None`

    See Also
    --------
    :func:`~regressions.analyses.mean_values.average_value_derivative`, :func:`~regressions.analyses.mean_values.mean_values_derivative`, :func:`~regressions.analyses.mean_values.average_value_integral`, :func:`~regressions.analyses.mean_values.mean_values_integral`

    Notes
    -----
    - |mean_derivatives|
    - |mean_integrals|

    Examples
    --------
    Import `average_values` function from `regressions` library
        >>> from regressions.analyses.mean_values import average_values
    Calculate the averages of a cubic function with coefficients 2, 3, 5, and 7 between 10 and 20
        >>> averages_cubic = average_values('cubic', [2, 3, 5, 7], 10, 20)
        >>> print(averages_cubic['average_value_derivative'])
        1495.0
        >>> print(averages_cubic['mean_values_derivative'])
        [15.2665]
        >>> print(averages_cubic['average_value_integral'])
        8282.0
        >>> print(averages_cubic['mean_values_integral'])
        [15.5188]
    Calculate the averages of a sinusoidal function with coefficients 2, 3, 5, and 7 between 10 and 20
        >>> averages_sinusoidal = average_values('sinusoidal', [2, 3, 5, 7], 10, 20)
        >>> print(averages_sinusoidal['average_value_derivative'])
        0.0401
        >>> print(averages_sinusoidal['mean_values_derivative'])
        [10.7618, 11.8046, 12.8562, 13.899, 14.9506, 15.9934, 17.045, 18.0878, 19.1394, '10.7618 + 2.0944k', '11.8046 + 2.0944k']
        >>> print(averages_sinusoidal['average_value_integral'])
        6.9143
        >>> print(averages_sinusoidal['mean_values_integral'])
        [10.2503, 11.2689, 12.3447, 13.3633, 14.4391, 15.4577, 16.5335, 17.5521, 18.6279, 19.6465, '10.2503 + 2.0944k', '11.2689 + 2.0944k']
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    compare_scalars(start, end, 'third', 'fourth')
    positive_integer(precision)

    # Determine various mean values
    derivative_value = average_value_derivative(equation_type, coefficients, start, end, precision)
    derivative_inputs = mean_values_derivative(equation_type, coefficients, start, end, precision)
    integral_value = average_value_integral(equation_type, coefficients, start, end, precision)
    integral_inputs = mean_values_integral(equation_type, coefficients, start, end, precision)

    # Package all values in single dictionary
    results = {
        'average_value_derivative': derivative_value,
        'mean_values_derivative': derivative_inputs,
        'average_value_integral': integral_value,
        'mean_values_integral': integral_inputs
    }
    return results