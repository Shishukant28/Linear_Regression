from regressions.errors.analyses import select_equations, select_points
from regressions.errors.scalars import scalar_value, compare_scalars, positive_integer
from regressions.errors.vectors import vector_of_scalars, allow_none_vector
from regressions.errors.matrices import allow_none_matrix, allow_vector_matrix
from regressions.statistics.rounding import rounded_value, rounded_list
from regressions.statistics.sort import sorted_list, sorted_strings
from regressions.statistics.ranges import shift_into_range
from regressions.vectors.unify import unite_vectors
from regressions.vectors.separate import separate_elements
from regressions.vectors.generate import generate_elements
from .equations.linear import linear_equation
from .equations.quadratic import quadratic_equation
from .equations.cubic import cubic_equation
from .equations.hyperbolic import hyperbolic_equation
from .equations.exponential import exponential_equation
from .equations.logarithmic import logarithmic_equation
from .equations.logistic import logistic_equation
from .equations.sinusoidal import sinusoidal_equation
from .intercepts import intercept_points
from .extrema import extrema_points
from .inflections import inflection_points

def coordinate_pairs(equation_type, coefficients, inputs, point_type = 'point', precision = 4):
    """
    Creates a list of coordinate pairs from a set of inputs

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which coordinate pairs must be determined (e.g., 'linear', 'quadratic')
    coefficients : list of int or float
        Coefficients to use to generate the equation to investigate
    inputs : list of int or float or str
        X-coordinates to use to generate the y-coordinates for each coordinate pair
    point_type : str, default='point'
        Name of the type of point that describes all points which must be generated (e.g., 'intercepts', 'maxima')
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the results

    Raises
    ------
    ValueError
        First argument must be either 'linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', or 'sinusoidal'
    TypeError
        Second argument must be a 1-dimensional list containing elements that are integers or floats
    TypeError
        Third argument must be a 1-dimensional list containing elements that are integers, floats, strings, or None
    ValueError
        Fourth argument must be either 'point', 'intercepts', 'maxima', 'minima', or 'inflections'
    ValueError
        Last argument must be a positive integer
        
    Returns
    -------
    points : list of float or str
        List containing lists of coordinate pairs, in which the second element of the inner lists are floats and the first elements of the inner lists are either floats or strings (the latter for general forms); may return a list of None if inputs list contained None

    See Also
    --------
    :func:`~regressions.vectors.generate.generate_elements`, :func:`~regressions.vectors.unify.unite_vectors`

    Notes
    -----
    - Set of x-coordinates of points: :math:`x_i = \\{ x_1, x_2, \\cdots, x_n \\}`
    - Set of y-coordinates of points: :math:`y_i = \\{ y_1, y_2, \\cdots, y_n \\}`
    - Set of coordinate pairs of points: :math:`p_i = \\{ (x_1, y_1), (x_2, y_2), \\cdots, (x_n, y_n) \\}`

    Examples
    --------
    Import `coordinate_pairs` function from `regressions` library
        >>> from regressions.analyses.points import coordinate_pairs
    Generate a list of coordinate pairs for a cubic function with coefficients 2, 3, 5, and 7 based off x-coordinates of 1, 2, 3, and 4
        >>> points_cubic = coordinate_pairs('cubic', [2, 3, 5, 7], [1, 2, 3, 4])
        >>> print(points_cubic)
        [[1.0, 17.0], [2.0, 45.0], [3.0, 103.0], [4.0, 203.0]]
    Generate a list of coordinate pairs for a sinusoidal function with coefficients 2, 3, 5, and 7 based off x-coordinates of 1, 2, 3, and 4
        >>> points_sinusoidal = coordinate_pairs('sinusoidal', [2, 3, 5, 7], [1, 2, 3, 4])
        >>> print(points_sinusoidal)
        [[1.0, 8.0731], [2.0, 6.1758], [3.0, 7.5588], [4.0, 6.7178]]
    Generate a list of coordinate pairs for a quadratic function with coefficients 1, -5, and 6 based off x-coordinates of 2 and 3 (given that the resultant coordinates will be x-intercepts)
        >>> points_quadratic = coordinate_pairs('quadratic', [1, -5, 6], [2, 3], 'intercepts')
        >>> print(points_quadratic)
        [[2.0, 0.0], [3.0, 0.0]]
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    allow_none_vector(inputs, 'third')
    select_points(point_type, 'fourth')
    positive_integer(precision)

    # Create equations for evaluating inputs (based on equation type)
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

    # Round inputs
    rounded_inputs = []
    for point in inputs:
        if isinstance(point, (int, float)):
            rounded_inputs.append(rounded_value(point, precision))
        else:
            rounded_inputs.append(point)

    # Create empty lists
    outputs = []
    coordinates = []
    
    # Handle no points
    if rounded_inputs[0] == None:
        coordinates.append(None)
    
    # Fill outputs list with output value at each input
    else:
        for value in rounded_inputs:
            # Circumvent inaccurate rounding
            if point_type == 'intercepts':
                outputs.append(0.0)
            
            # Evaluate function at inputs
            else:
                # Evaluate numerical inputs
                if isinstance(value, (int, float)):
                    output = equation(value)
                    rounded_output = rounded_value(output, precision)
                    outputs.append(rounded_output)
                
                # Handle non-numerical inputs
                else:
                    outputs.append(outputs[0])

        # Unite inputs and outputs for maxima into single list
        coordinates.extend(unite_vectors(rounded_inputs, outputs))
    
    # Return final coordinate pairs
    return coordinates

def key_coordinates(equation_type, coefficients, precision = 4):
    """
    Calculates the key points of a specific function

    Parameters
    ----------
    equation_type : str
        Name of the type of function for which key points must be determined (e.g., 'linear', 'quadratic')
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
    points['roots'] : list of float or str
        List containing two-element lists for each point; first elements of those lists will be the value of the x-coordinate at which the original function has a root; second elements of those lists will be 0; if the function is sinusoidal, then only the initial results within a four-period interval will be listed, but general forms will also be included; if the function has no roots, then it will return a list of `None`
    points['maxima'] : list of float or str
        List containing two-element lists for each point; first elements of those lists will be the value of the x-coordinate at which the original function has a relative maximum; second elements of those lists will be the y-coordinate of that maximum; if the function is sinusoidal, then only the initial results within a two-period interval will be listed, but a general form will also be included; if the function has no maxima, then it will return a list of `None`
    points['minima'] : list of float or str
        List containing two-element lists for each point; first elements of those lists will be the value of the x-coordinate at which the original function has a relative minimum; second elements of those lists will be the y-coordinate of that minimum; if the function is sinusoidal, then only the initial results within a two-period interval will be listed, but a general form will also be included; if the function has no minima, then it will return a list of `None`
    points['inflections'] : list of float or str
        List containing two-element lists for each point; first elements of those lists will be the value of the x-coordinate at which the original function has an inflection; second elements of those lists will be the y-coordinate of that inflection; if the function is sinusoidal, then only the initial results within a two-period interval will be listed, but a general form will also be included; if the function has no inflection points, then it will return a list of `None`

    See Also
    --------
    - Roots for key functions: :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.analyses.roots.quadratic.quadratic_roots`, :func:`~regressions.analyses.roots.cubic.cubic_roots`, :func:`~regressions.analyses.roots.hyperbolic.hyperbolic_roots`, :func:`~regressions.analyses.roots.exponential.exponential_roots`, :func:`~regressions.analyses.roots.logarithmic.logarithmic_roots`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.analyses.roots.sinusoidal.sinusoidal_roots`
    - Graphical analysis: :func:`~regressions.analyses.criticals.critical_points`, :func:`~regressions.analyses.intervals.sign_chart`, :func:`~regressions.analyses.maxima.maxima_points`, :func:`~regressions.analyses.minima.minima_points`, :func:`~regressions.analyses.extrema.extrema_points`, :func:`~regressions.analyses.inflections.inflection_points`

    Notes
    -----
    - Key points include x-intercepts, maxima, minima, and points of inflection
    - |intercepts|
    - |extrema|
    - |inflections|

    Examples
    --------
    Import `key_coordinates` function from `regressions` library
        >>> from regressions.analyses.points import key_coordinates
    Calculate the key points of a cubic function with coefficients 1, -15, 63, and -7
        >>> points_cubic = key_coordinates('cubic', [1, -15, 63, -7])
        >>> print(points_cubic['roots'])
        [[0.1142, 0.0]]
        >>> print(points_cubic['maxima'])
        [[3.0, 74.0]]
        >>> print(points_cubic['minima'])
        [[7.0, 42.0]]
        >>> print(points_cubic['inflections'])
        [[5.0, 58.0]]
    Calculate the key points of a sinusoidal function with coefficients 2, 3, 5, and 1
        >>> points_sinusoidal = key_coordinates('sinusoidal', [2, 3, 5, 1])
        >>> print(points_sinusoidal['roots'])
        [[4.8255, 0.0], [6.2217, 0.0], [6.9199, 0.0], [8.3161, 0.0], [9.0143, 0.0], [10.4105, 0.0], [11.1087, 0.0], [12.5049, 0.0], [13.203, 0.0], [14.5993, 0.0], ['4.8255 + 2.0944k', 0.0], ['6.2217 + 2.0944k', 0.0]]
        >>> print(points_sinusoidal['maxima'])
        [[5.5236, 3.0], [7.618, 3.0], [9.7124, 3.0], ['5.5236 + 2.0944k', 3.0]]
        >>> print(points_sinusoidal['minima'])
        [[6.5708, -1.0], [8.6652, -1.0], ['6.5708 + 2.0944k', -1.0]]
        >>> print(points_sinusoidal['inflections'])
        [[5.0, 1.0], [6.0472, 1.0], [7.0944, 1.0], [8.1416, 1.0], [9.1888, 1.0001], ['5.0 + 1.0472k', 1.0]]
    """
    # Handle input errors
    select_equations(equation_type)
    vector_of_scalars(coefficients, 'second')
    positive_integer(precision)
    
    # Create lists of inputs
    intercepts_inputs = intercept_points(equation_type, coefficients, precision)
    extrema_inputs = extrema_points(equation_type, coefficients, precision)
    maxima_inputs = extrema_inputs['maxima']
    minima_inputs = extrema_inputs['minima']
    inflections_inputs = inflection_points(equation_type, coefficients, precision)

    # Generate coordinate pairs for all x-intercepts
    intercepts_coordinates = coordinate_pairs(equation_type, coefficients, intercepts_inputs, 'intercepts', precision)
    
    # Generate coordinate pairs for all maxima
    maxima_coordinates = coordinate_pairs(equation_type, coefficients, maxima_inputs, 'maxima', precision)
    
    # Generate coordinate pairs for all minima
    minima_coordinates = coordinate_pairs(equation_type, coefficients, minima_inputs, 'minima', precision)
    
    # Generate coordinate pairs for all points of inflection
    inflections_coordinates = coordinate_pairs(equation_type, coefficients, inflections_inputs, 'inflections', precision)
    
    # Create dictionary to return
    result = {
        'roots': intercepts_coordinates,
        'maxima': maxima_coordinates,
        'minima': minima_coordinates,
        'inflections': inflections_coordinates
    }
    return result

def points_within_range(points, start, end):
    """
    Eliminates all values from a set of points that fall below a lower bound or above an upper bound

    Parameters
    ----------
    points : list of int or float or str
        Set of points to narrow down to only those within a certain range
    start : int or float
        Lower bound of range into which the initial value must be adjusted (final value should be greater than or equal to start)
    end : int or float
        Upper bound of range into which the initial value must be adjusted (final value should be less than or equal to end)

    Raises
    ------
    TypeError
        First argument must be a 1-dimensional list containing elements that are integers, floats, strings, or None
    TypeError
        Second and third arguments must be integers or floats
    ValueError
        Second argument must be less than or equal to third argument

    Returns
    -------
    selected_points : list of int or float or str
        List of all values from original list that fall within specified range; may return a list of None if no points from the original list fall within range or if original list only contained None

    See Also
    --------
    :func:`~regressions.vectors.separate.separate_elements`, :func:`~regressions.statistics.ranges.shift_into_range`, :func:`~regressions.analyses.mean_values.mean_values_derivative`, :func:`~regressions.analyses.mean_values.mean_values_integral`

    Notes
    -----
    - Initial set of points: :math:`p_i = \\{ p_1, p_2, \\cdots, p_n \\}`
    - Lower bound of range: :math:`b_l`
    - Upper bound of range: :math:`b_u`
    - Adjusted set of points within range: :math:`r_i = \\{ r \\mid r \\in p_i, r \\geq b_l, r \\leq b_u \\}`

    Examples
    --------
    Import `points_within_range` function from `regressions` library
        >>> from regressions.analyses.points import points_within_range
    Eliminate all points above 19 or below 6 in the set [1, 5, 6, 7, 18, 20, 50, 127]
        >>> selected_points_int = points_within_range([1, 5, 6, 7, 18, 20, 50, 127], 6, 19)
        >>> print(selected_points_int)
        [6, 7, 18]
    Eliminate all points above 243.7821 or below 198.1735 in the set [542.1234, 237.9109, -129.3214, 199.4321, 129.3214]
        >>> selected_points_float = points_within_range([542.1234, 237.9109, -129.3214, 199.4321, 129.3214], 198.1735, 243.7821)
        >>> print(selected_points_float)
        [237.9109, 199.4321]
    """
    # Handle input errors
    allow_none_vector(points, 'first')
    compare_scalars(start, end, 'second', 'third')

    # Separate numerical results from string results
    separated_results = separate_elements(points)
    numerical_results = separated_results['numerical']
    other_results = separated_results['other']

    # Eliminate numerical results outside of range
    selected_results = [x for x in numerical_results if x >= start and x <= end]

    # Create list to return
    final_results = []

    # Handle no results
    if not selected_results and not other_results:
        final_results.append(None)
    
    # Handle general case
    else:
        final_results.extend(selected_results + other_results)
    
    # Return results
    return final_results

def shifted_points_within_range(points, minimum, maximum, precision = 4):
    # Handle input errors
    allow_vector_matrix(points, 'first')
    compare_scalars(minimum, maximum, 'second', 'third')
    positive_integer(precision)

    # Grab general points
    general_points = []
    for point in points:
        # Handle coordinate pairs
        if isinstance(point, list):
            if isinstance(point[0], str):
                general_points.append(point[0])
        
        # Handle single coordinates
        else:
            if isinstance(point, str):
                general_points.append(point)
    
    # Generate options for inputs
    optional_points = []
    for point in general_points:
        # Grab initial value and periodic unit
        initial_value_index = point.find(' + ')
        initial_value = float(point[:initial_value_index])
        periodic_unit_index = initial_value_index + 3
        periodic_unit = float(point[periodic_unit_index:-1])
        
        # Increase or decrease initial value to fit into range
        alternative_initial_value = shift_into_range(initial_value, periodic_unit, minimum, maximum)
        
        # Generate additional values within range
        generated_elements = generate_elements(alternative_initial_value, periodic_unit, precision)
        optional_points += generated_elements
    
    # Separate numerical inputs from string inputs
    separated_points = separate_elements(optional_points)
    numerical_points = separated_points['numerical']
    other_points = separated_points['other']

    # Sort numerical inputs
    sorted_points = sorted_list(numerical_points)

    # Reduce numerical inputs to within a given range
    selected_points = [x for x in sorted_points if x >= minimum and x <= maximum]
    
    # Round numerical inputs
    rounded_points = rounded_list(selected_points, precision)
    
    # Sort string inputs
    sorted_other_points = sorted_strings(other_points)
    
    # Combine numerical and string inputs
    result = rounded_points + sorted_other_points
    return result

def shifted_coordinates_within_range(coordinates, minimum, maximum, precision = 4):
    # Handle input errors
    allow_none_matrix(coordinates, 'first')
    compare_scalars(minimum, maximum, 'second', 'third')
    positive_integer(precision)

    # Create list to return
    result = []

    # Handle general case
    if coordinates[0] is not None:
        # Generate inputs
        input_points = shifted_points_within_range(coordinates, minimum, maximum, precision)
        
        # Generate outputs
        output_points = []
        for point in input_points:
            output_points.append(coordinates[0][1])
        
        # Unite inputs and outputs into single list
        result = unite_vectors(input_points, output_points)
    
    # Handle no points
    else:
        result = coordinates
    
    # Return result
    return result