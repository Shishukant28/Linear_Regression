from regressions.errors.analyses import select_equations
from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import select_integers, positive_integer
from regressions.statistics.sort import sorted_list
from regressions.vectors.separate import separate_elements
from .derivatives.linear import linear_derivatives
from .derivatives.quadratic import quadratic_derivatives
from .derivatives.cubic import cubic_derivatives
from .derivatives.hyperbolic import hyperbolic_derivatives
from .derivatives.exponential import exponential_derivatives
from .derivatives.logarithmic import logarithmic_derivatives
from .derivatives.logistic import logistic_derivatives
from .derivatives.sinusoidal import sinusoidal_derivatives
from .criticals import critical_points

def sign_chart(equation_type, coefficients, derivative_level, precision = 4):
    """
    Creates a sign chart for a given derivative

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which the sign chart must be constructed (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients to use to generate the equation to investigate
    derivative_level : int
        Integer corresponding to which derivative to investigate for sign chart (1 for the first derivative and 2 for the second derivative)
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
    chart : list of str and float
        Strings describing the sign (e.g., 'positive', 'negative') of the derivative between its critical points; as a result, its elements will alternate between strings (indicating the signs) and floats (indicating the end points); if the function is sinusoidal, then only the initial results within a two-period interval will be listed, but a general form to determine other end points will also be included

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Graphical analysis: :func:`~regressions.analyses.criticals.critical_points`, :func:`~regressions.analyses.points.key_coordinates`

    Notes
    -----
    - Critical points for the derivative of a function: :math:`c_i = \\{ c_1, c_2, c_3,  \\cdots, c_{n-1}, c_n \\}`
    - Midpoints and key values of the intervals demarcated by the critical points: :math:`m_i = \\{ c_1 - 1,  \\frac{c_1+c_2}{2}, \\frac{c_2+c_3}{2}, \\cdots, \\frac{c_{n-1}+c_n}{2}, c_n + 1 \\}`
    - Values of the derivative within the intervals: :math:`v_i = \\{ v_1, v_2, v_3, v_4 \\cdots, v_{n-1}, v_n, v_{n+1} \\}`
    - Sign chart: :math:`s = ( v_1, c_1, v_2, c_2, v_3, c_3, v_4, \\dots, v_{n-1}, c_{n-1}, v_n, c_n, v_{n+1} )`

        - :math:`v_j = negative` if :math:`f'(m_j) < 0`
        - :math:`v_j = constant` if :math:`f'(m_j) = 0`
        - :math:`v_j = positve` if :math:`f'(m_j) > 0`

    - |intervals|

    Examples
    --------
    Import `sign_chart` function from `regressions` library
        >>> from regressions.analyses.intervals import sign_chart
    Create the sign chart for the first derivative of a cubic function with coefficients 1, -15, 63, and -7
        >>> chart_cubic = sign_chart('cubic', [1, -15, 63, -7], 1)
        >>> print(chart_cubic)
        ['positive', 3.0, 'negative', 7.0, 'positive']
    Create the sign chart for the second derivative of a sinusoidal function with coefficients 2, 3, 5, and 7
        >>> chart_sinusoidal = sign_chart('sinusoidal', [2, 3, 5, 7], 2)
        >>> print(chart_sinusoidal)
        ['positive', 5.0, 'negative', 6.0472, 'positive', 7.0944, 'negative', 8.1416, 'positive', 9.1888, 'negative', '5.0 + 1.0472k']
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    select_integers(derivative_level, [1, 2], 'third')
    positive_integer(precision)

    # Create first and second derivatives based on equation type
    both_derivatives = {}
    if equation_type == 'linear':
        both_derivatives = linear_derivatives(*coefficients, precision)
    elif equation_type == 'quadratic':
        both_derivatives = quadratic_derivatives(*coefficients, precision)
    elif equation_type == 'cubic':
        both_derivatives = cubic_derivatives(*coefficients, precision)
    elif equation_type == 'hyperbolic':
        both_derivatives = hyperbolic_derivatives(*coefficients, precision)
    elif equation_type == 'exponential':
        both_derivatives = exponential_derivatives(*coefficients, precision)
    elif equation_type == 'logarithmic':
        both_derivatives = logarithmic_derivatives(*coefficients, precision)
    elif equation_type == 'logistic':
        both_derivatives = logistic_derivatives(*coefficients, precision)
    elif equation_type == 'sinusoidal':
        both_derivatives = sinusoidal_derivatives(*coefficients, precision)
    
    # Grab specific derivative evaluation based on derivative level
    derivative = lambda x : x
    if derivative_level == 1:
        derivative = both_derivatives['first']['evaluation']
    elif derivative_level == 2:
        derivative = both_derivatives['second']['evaluation']
    
    # Create critical points for specific derivative
    points = critical_points(equation_type, coefficients, derivative_level, precision)
    
    # Create sign chart for specific derivative
    result = []

    # Handle no critical points
    if points[0] == None:
        # Test an arbitrary number
        if derivative(10) > 0:
            result = ['positive']
        elif derivative(10) < 0:
            result = ['negative']
        else:
            result = ['constant']
    
    # Handle exactly one critical point
    elif len(points) == 1:
        # Generate numbers to test
        turning_point = points[0]
        before = turning_point - 1
        after = turning_point + 1

        # Test numbers
        if derivative(before) > 0:
            before = 'positive'
        elif derivative(before) < 0:
            before = 'negative'
        if derivative(after) > 0 :
            after = 'positive'
        elif derivative(after) < 0:
            after = 'negative'
        
        # Store sign chart
        result = [before, turning_point, after]
    
    # Handle exactly two critical points
    elif len(points) == 2:
        # Generate numbers to test
        sorted_points = sorted_list(points)
        first_point = sorted_points[0]
        second_point = sorted_points[1]
        middle = (first_point + second_point) / 2
        before = first_point - 1
        after = second_point + 1

        # Test numbers
        if derivative(before) > 0:
            before = 'positive'
        elif derivative(before) < 0:
            before = 'negative'
        if derivative(middle) > 0 :
            middle = 'positive'
        elif derivative(middle) < 0:
            middle = 'negative'
        if derivative(after) > 0:
            after = 'positive'
        elif derivative(after) < 0:
            after = 'negative'
        
        # Store sign chart
        result = [before, first_point, middle, second_point, after]
    
    # Handle more than two critical points
    else:
        # Separate numerical inputs from string inputs
        separated_points = separate_elements(points)
        numerical_points = separated_points['numerical']
        other_points = separated_points['other']
        
        # Generate numbers to test
        sorted_points = sorted_list(numerical_points)
        difference = sorted_points[1] - sorted_points[0]
        halved_difference = difference / 2
        before_first = sorted_points[0] - halved_difference
        between_first_second = sorted_points[0] + halved_difference
        between_second_third = sorted_points[1] + halved_difference
        between_third_fourth = sorted_points[2] + halved_difference
        between_fourth_last = sorted_points[3] + halved_difference
        after_last = sorted_points[4] + halved_difference
        test_points = [before_first, between_first_second, between_second_third, between_third_fourth, between_fourth_last, after_last]

        # Test numbers
        signs = []
        for point in test_points:
            if derivative(point) > 0:
                signs.append('positive')
            elif derivative(point) < 0:
                signs.append('negative')
        
        # Store sign chart
        result = [signs[0], sorted_points[0], signs[1], sorted_points[1], signs[2], sorted_points[2], signs[3], sorted_points[3], signs[4], sorted_points[4], signs[5], *other_points]
    return result