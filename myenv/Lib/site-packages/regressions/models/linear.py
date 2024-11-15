from regressions.errors.matrices import matrix_of_scalars
from regressions.errors.vectors import long_vector
from regressions.errors.scalars import positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.vectors.dimension import single_dimension
from regressions.vectors.column import column_conversion
from regressions.matrices.solve import system_solution
from regressions.analyses.equations.linear import linear_equation
from regressions.analyses.derivatives.linear import linear_derivatives
from regressions.analyses.integrals.linear import linear_integral
from regressions.analyses.points import key_coordinates
from regressions.analyses.accumulation import accumulated_area
from regressions.analyses.mean_values import average_values
from regressions.statistics.summary import five_number_summary
from regressions.statistics.correlation import correlation_coefficient

def linear_model(data, precision = 4):
    """
    Generates a linear regression model from a given data set

    Parameters
    ----------
    data : list of lists of int or float
        List of lists of numbers representing a collection of coordinate pairs; it must include at least 10 pairs
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the results

    Raises
    ------
    TypeError
        First argument must be a 2-dimensional list
    TypeError
        Elements nested within first argument must be integers or floats
    ValueError
        First argument must contain at least 10 elements
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    model['constants'] : list of float
        Coefficients of the resultant linear model; the first element is the coefficient of the linear term, and the second element is the coefficient of the constant term
    model['evaluations']['equation'] : func
        Function that evaluates the equation of the linear model at a given numeric input (e.g., model['evaluations']['equation'](10) would evaluate the equation of the linear model when the independent variable is 10)
    model['evaluations']['derivative'] : func
        Function that evaluates the first derivative of the linear model at a given numeric input (e.g., model['evaluations']['derivative'](10) would evaluate the first derivative of the linear model when the independent variable is 10)
    model['evaluations']['integral'] : func
        Function that evaluates the integral of the linear model at a given numeric input (e.g., model['evaluations']['integral'](10) would evaluate the integral of the linear model when the independent variable is 10)
    model['points']['roots'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the x-intercepts of the linear model (will contain exactly one point)
    model['points']['maxima'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the maxima of the linear model (will always be `None`)
    model['points']['minima'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the minima of the linear model (will always be `None`)
    model['points']['inflections'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the inflection points of the linear model (will always be `None`)
    model['accumulations']['range'] : float
        Total area under the curve represented by the linear model between the smallest independent coordinate originally provided and the largest independent coordinate originally provided (i.e., over the range)
    model['accumulations']['iqr'] : float
        Total area under the curve represented by the linear model between the first and third quartiles of all the independent coordinates originally provided (i.e., over the interquartile range)
    model['averages']['range']['average_value_derivative'] : float
        Average rate of change of the curve represented by the linear model between the smallest independent coordinate originally provided and the largest independent coordinate originally provided
    model['averages']['range']['mean_values_derivative'] : list of float
        All points between the smallest independent coordinate originally provided and the largest independent coordinate originally provided where their instantaneous rate of change equals the function's average rate of change over that interval
    model['averages']['range']['average_value_integral'] : float
        Average value of the curve represented by the linear model between the smallest independent coordinate originally provided and the largest independent coordinate originally provided
    model['averages']['range']['mean_values_integral'] : list of float
        All points between the smallest independent coordinate originally provided and the largest independent coordinate originally provided where their value equals the function's average value over that interval
    model['averages']['iqr']['average_value_derivative'] : float
        Average rate of change of the curve represented by the linear model between the first and third quartiles of all the independent coordinates originally provided
    model['averages']['iqr']['mean_values_derivative'] : list of float
        All points between the first and third quartiles of all the independent coordinates originally provided where their instantaneous rate of change equals the function's average rate of change over that interval
    model['averages']['iqr']['average_value_integral'] : float
        Average value of the curve represented by the linear model between the first and third quartiles of all the independent coordinates originally provided
    model['averages']['iqr']['mean_values_integral'] : list of float
        All points between the first and third quartiles of all the independent coordinates originally provided where their value equals the function's average value over that interval
    model['correlation'] : float
        Correlation coefficient indicating how well the model fits the original data set (values range between 0.0, implying no fit, and 1.0, implying a perfect fit)

    See Also
    --------
    :func:`~regressions.analyses.equations.linear.linear_equation`, :func:`~regressions.analyses.derivatives.linear.linear_derivatives`, :func:`~regressions.analyses.integrals.linear.linear_integral`, :func:`~regressions.analyses.roots.linear.linear_roots`, :func:`~regressions.statistics.correlation.correlation_coefficient`, :func:`~regressions.execute.run_all`

    Notes
    -----
    - Provided ordered pairs for the data set: :math:`p_i = \\{ (p_{1,x}, p_{1,y}), (p_{2,x}, p_{2,y}), \\cdots, (p_{n,x}, p_{n,y}) \\}`
    - Provided values for the independent variable: :math:`X_i = \\{ p_{1,x}, p_{2,x}, \\cdots, p_{n,x} \\}`
    - Provided values for the dependent variable: :math:`Y_i = \\{ p_{1,y}, p_{2,y}, \\cdots, p_{n,y} \\}`
    - Minimum value of the provided values for the independent variable: :math:`X_{min} \\leq p_{j,x}, \\forall p_{j,x} \\in X_i`
    - Maximum value of the provided values for the independent variable: :math:`X_{max} \\geq p_{j,x}, \\forall p_{j,x} \\in X_i`
    - First quartile of the provided values for the independent variable: :math:`X_{Q1}`
    - Third quartile of the provided values for the independent variable: :math:`X_{Q3}`
    - Mean of all provided values for the dependent variable: :math:`\\bar{y} = \\frac{1}{n}\\cdot{\\sum\\limits_{i=1}^n Y_i}`
    - Resultant values for the coefficients of the linear model: :math:`C_i = \\{ a, b \\}`
    - Standard form for the equation of the linear model: :math:`f(x) = a\\cdot{x} + b`
    - First derivative of the linear model: :math:`f'(x) = a`
    - Second derivative of the linear model: :math:`f''(x) = 0`
    - Integral of the linear model: :math:`F(x) = \\frac{a}{2}\\cdot{x^2} + b\\cdot{x}`
    - Potential x-values of the roots of the linear model: :math:`x_{intercepts} = \\{ -\\frac{b}{a} \\}`
    - Potential x-values of the maxima of the linear model: :math:`x_{maxima} = \\{ \\varnothing \\}`
    - Potential x-values of the minima of the linear model: :math:`x_{minima} = \\{ \\varnothing \\}`
    - Potential x-values of the inflection points of the linear model: :math:`x_{inflections} = \\{ \\varnothing \\}`
    - Accumulatation of the linear model over its range: :math:`A_{range} = \\int_{X_{min}}^{X_{max}} f(x) \\,dx`
    - Accumulatation of the linear model over its interquartile range: :math:`A_{iqr} = \\int_{X_{Q1}}^{X_{Q3}} f(x) \\,dx`
    - Average rate of change of the linear model over its range: :math:`m_{range} = \\frac{f(X_{max}) - f(X_{min})}{X_{max} - X_{min}}`
    - Potential x-values at which the linear model's instantaneous rate of change equals its average rate of change over its range: :math:`x_{m,range} = \\{ [X_{min}, X_{max}] \\}`
    - Average value of the linear model over its range: :math:`v_{range} = \\frac{1}{X_{max} - X_{min}}\\cdot{A_{range}}`
    - Potential x-values at which the linear model's value equals its average value over its range: :math:`x_{v,range} = \\{ -\\frac{b - v_{range}}{a} \\}`
    - Average rate of change of the linear model over its interquartile range: :math:`m_{iqr} = \\frac{f(X_{Q3}) - f(X_{Q1})}{X_{Q3} - X_{Q1}}`
    - Potential x-values at which the linear model's instantaneous rate of change equals its average rate of change over its interquartile range: :math:`x_{m,iqr} = \\{ [X_{Q1}, X_{Q3}] \\}`
    - Average value of the linear model over its interquartile range: :math:`v_{iqr} = \\frac{1}{X_{Q3} - X_{Q1}}\\cdot{A_{iqr}}`
    - Potential x-values at which the linear model's value equals its average value over its interquartile range: :math:`x_{v,iqr} = \\{ -\\frac{b - v_{iqr}}{a} \\}`
    - Predicted values based on the linear model: :math:`\\hat{y}_i = \\{ \\hat{y}_1, \\hat{y}_2, \\cdots, \\hat{y}_n \\}`
    - Residuals of the dependent variable: :math:`e_i = \\{ p_{1,y} - \\hat{y}_1, p_{2,y} - \\hat{y}_2, \\cdots, p_{n,y} - \\hat{y}_n \\}`
    - Deviations of the dependent variable: :math:`d_i = \\{ p_{1,y} - \\bar{y}, p_{2,y} - \\bar{y}, \\cdots, p_{n,y} - \\bar{y} \\}`
    - Sum of squares of residuals: :math:`SS_{res} = \\sum\\limits_{i=1}^n e_i^2`
    - Sum of squares of deviations: :math:`SS_{dev} = \\sum\\limits_{i=1}^n d_i^2`
    - Correlation coefficient for the linear model: :math:`r = \\sqrt{1 - \\frac{SS_{res}}{SS_{dev}}}`
    - |regression_analysis|

    Examples
    --------
    Import `linear_model` function from `regressions` library
        >>> from regressions.models.linear import linear_model
    Generate a linear regression model for the data set [[1, 30], [2, 27], [3, 24], [4, 21], [5, 18], [6, 15], [7, 12], [8, 9], [9, 6], [10, 3]], then print its coefficients, roots, total accumulation over its interquartile range, and correlation
        >>> model_perfect = linear_model([[1, 30], [2, 27], [3, 24], [4, 21], [5, 18], [6, 15], [7, 12], [8, 9], [9, 6], [10, 3]])
        >>> print(model_perfect['constants'])
        [-3.0, 33.0]
        >>> print(model_perfect['points']['roots'])
        [[11.0, 0.0]]
        >>> print(model_perfect['accumulations']['iqr'])
        82.5
        >>> print(model_perfect['correlation'])
        1.0
    Generate a linear regression model for the data set [[1, 32], [2, 25], [3, 14], [4, 23], [5, 39], [6, 45], [7, 42], [8, 49], [9, 36], [10, 33]], then print its coefficients, inflections, total accumulation over its range, and correlation
        >>> model_agnostic = linear_model([[1, 32], [2, 25], [3, 14], [4, 23], [5, 39], [6, 45], [7, 42], [8, 49], [9, 36], [10, 33]])
        >>> print(model_agnostic['constants'])
        [1.9636, 23.0]
        >>> print(model_agnostic['points']['inflections'])
        [None]
        >>> print(model_agnostic['accumulations']['range'])
        304.1982
        >>> print(model_agnostic['correlation'])
        0.5516
    """
    # Handle input errors
    matrix_of_scalars(data, 'first')
    long_vector(data)
    positive_integer(precision)

    # Store independent and dependent variable values separately
    independent_variable = single_dimension(data, 1)
    dependent_variable = single_dimension(data, 2)

    # Create matrices for independent and dependent variables
    independent_matrix = []
    dependent_matrix = column_conversion(dependent_variable)

    # Iterate over inputted data
    for element in independent_variable:
        # Store linear and constant evaluations of original independent elements together as lists within independent matrix
        independent_matrix.append([element, 1])

    # Solve system of equations
    solution = system_solution(independent_matrix, dependent_matrix, precision)

    # Eliminate zeroes from solution
    coefficients = no_zeroes(solution, precision)

    # Generate evaluations for function, derivatives, and integral
    equation = linear_equation(*coefficients, precision)
    derivative = linear_derivatives(*coefficients, precision)['first']['evaluation']
    integral = linear_integral(*coefficients, precision)['evaluation']

    # Determine key points of graph
    points = key_coordinates('linear', coefficients, precision)

    # Generate values for lower and upper bounds
    five_numbers = five_number_summary(independent_variable, precision)
    min_value = five_numbers['minimum']
    max_value = five_numbers['maximum']
    q1 = five_numbers['q1']
    q3 = five_numbers['q3']

    # Calculate accumulations
    accumulated_range = accumulated_area('linear', coefficients, min_value, max_value, precision)
    accumulated_iqr = accumulated_area('linear', coefficients, q1, q3, precision)

    # Determine average values and their points
    averages_range = average_values('linear', coefficients, min_value, max_value, precision)
    averages_iqr = average_values('linear', coefficients, q1, q3, precision)

    # Create list of predicted outputs
    predicted = []
    for element in independent_variable:
        predicted.append(equation(element))
    
    # Calculate correlation coefficient for model
    accuracy = correlation_coefficient(dependent_variable, predicted, precision)

    # Package preceding results in multiple dictionaries
    evaluations = {
        'equation': equation,
        'derivative': derivative,
        'integral': integral
    }
    points = {
        'roots': points['roots'],
        'maxima': points['maxima'],
        'minima': points['minima'],
        'inflections': points['inflections']
    }
    accumulations = {
        'range': accumulated_range,
        'iqr': accumulated_iqr
    }
    averages = {
        'range': averages_range,
        'iqr': averages_iqr
    }

    # Package all dictionaries in single dictionary to return
    result = {
        'constants': coefficients,
        'evaluations': evaluations,
        'points': points,
        'accumulations': accumulations,
        'averages': averages,
        'correlation': accuracy
    }
    return result