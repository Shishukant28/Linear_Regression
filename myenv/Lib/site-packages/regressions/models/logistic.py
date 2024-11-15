from numpy import exp, inf
from scipy.optimize import curve_fit
from regressions.errors.matrices import matrix_of_scalars
from regressions.errors.vectors import long_vector
from regressions.errors.scalars import positive_integer
from regressions.errors.adjustments import no_zeroes
from regressions.vectors.dimension import single_dimension
from regressions.analyses.equations.logistic import logistic_equation
from regressions.analyses.derivatives.logistic import logistic_derivatives
from regressions.analyses.integrals.logistic import logistic_integral
from regressions.analyses.points import key_coordinates
from regressions.analyses.accumulation import accumulated_area
from regressions.analyses.mean_values import average_values
from regressions.statistics.halve import half_dimension
from regressions.statistics.mean import mean_value
from regressions.statistics.summary import five_number_summary
from regressions.statistics.correlation import correlation_coefficient
from regressions.statistics.rounding import rounded_value

def logistic_model(data, precision = 4):
    """
    Generates a logistic regression model from a given data set

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
        Coefficients of the resultant logistic model; the first element is the carrying capacity, the second element is the growth rate, and the third element is the sigmoid's midpoint
    model['evaluations']['equation'] : func
        Function that evaluates the equation of the logistic model at a given numeric input (e.g., model['evaluations']['equation'](10) would evaluate the equation of the logistic model when the independent variable is 10)
    model['evaluations']['derivative'] : func
        Function that evaluates the first derivative of the logistic model at a given numeric input (e.g., model['evaluations']['derivative'](10) would evaluate the first derivative of the logistic model when the independent variable is 10)
    model['evaluations']['integral'] : func
        Function that evaluates the integral of the logistic model at a given numeric input (e.g., model['evaluations']['integral'](10) would evaluate the integral of the logistic model when the independent variable is 10)
    model['points']['roots'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the x-intercepts of the logistic model (will always be `None`)
    model['points']['maxima'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the maxima of the logistic model (will always be `None`)
    model['points']['minima'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the minima of the logistic model (will always be `None`)
    model['points']['inflections'] : list of lists of float
        List of lists of numbers representing the coordinate pairs of all the inflection points of the logistic model (will contain exactly one point)
    model['accumulations']['range'] : float
        Total area under the curve represented by the logistic model between the smallest independent coordinate originally provided and the largest independent coordinate originally provided (i.e., over the range)
    model['accumulations']['iqr'] : float
        Total area under the curve represented by the logistic model between the first and third quartiles of all the independent coordinates originally provided (i.e., over the interquartile range)
    model['averages']['range']['average_value_derivative'] : float
        Average rate of change of the curve represented by the logistic model between the smallest independent coordinate originally provided and the largest independent coordinate originally provided
    model['averages']['range']['mean_values_derivative'] : list of float
        All points between the smallest independent coordinate originally provided and the largest independent coordinate originally provided where their instantaneous rate of change equals the function's average rate of change over that interval
    model['averages']['range']['average_value_integral'] : float
        Average value of the curve represented by the logistic model between the smallest independent coordinate originally provided and the largest independent coordinate originally provided
    model['averages']['range']['mean_values_integral'] : list of float
        All points between the smallest independent coordinate originally provided and the largest independent coordinate originally provided where their value equals the function's average value over that interval
    model['averages']['iqr']['average_value_derivative'] : float
        Average rate of change of the curve represented by the logistic model between the first and third quartiles of all the independent coordinates originally provided
    model['averages']['iqr']['mean_values_derivative'] : list of float
        All points between the first and third quartiles of all the independent coordinates originally provided where their instantaneous rate of change equals the function's average rate of change over that interval
    model['averages']['iqr']['average_value_integral'] : float
        Average value of the curve represented by the logistic model between the first and third quartiles of all the independent coordinates originally provided
    model['averages']['iqr']['mean_values_integral'] : list of float
        All points between the first and third quartiles of all the independent coordinates originally provided where their value equals the function's average value over that interval
    model['correlation'] : float
        Correlation coefficient indicating how well the model fits the original data set (values range between 0.0, implying no fit, and 1.0, implying a perfect fit)

    See Also
    --------
    :func:`~regressions.analyses.equations.logistic.logistic_equation`, :func:`~regressions.analyses.derivatives.logistic.logistic_derivatives`, :func:`~regressions.analyses.integrals.logistic.logistic_integral`, :func:`~regressions.analyses.roots.logistic.logistic_roots`, :func:`~regressions.statistics.correlation.correlation_coefficient`, :func:`~regressions.execute.run_all`

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
    - Resultant values for the coefficients of the logistic model: :math:`C_i = \\{ a, b, c \\}`
    - Standard form for the equation of the logistic model: :math:`f(x) = \\frac{a}{1 + \\text{e}^{-b\\cdot(x - c)}}`
    - First derivative of the logistic model: :math:`f'(x) = \\frac{ab\\cdot{\\text{e}^{-b\\cdot(x - c)}}}{(1 + \\text{e}^{-b\\cdot(x - c)})^2}`
    - Second derivative of the logistic model: :math:`f''(x) = \\frac{2ab^2\\cdot{\\text{e}^{-2b\\cdot(x - c)}}}{(1 + \\text{e}^{-b\\cdot(x - c)})^3} - \\frac{ab^2\\cdot{\\text{e}^{-b\\cdot(x - c)}}}{(1 + \\text{e}^{-b\\cdot(x - c)})^2}`
    - Integral of the logistic model: :math:`F(x) = \\frac{a}{b}\\cdot{\\ln|\\text{e}^{b\\cdot(x - c)} + 1|}`
    - Potential x-values of the roots of the logistic model: :math:`x_{intercepts} = \\{ \\varnothing \\}`
    - Potential x-values of the maxima of the logistic model: :math:`x_{maxima} = \\{ \\varnothing \\}`
    - Potential x-values of the minima of the logistic model: :math:`x_{minima} = \\{ \\varnothing \\}`
    - Potential x-values of the inflection points of the logistic model: :math:`x_{inflections} = \\{ c \\}`
    - Accumulatation of the logistic model over its range: :math:`A_{range} = \\int_{X_{min}}^{X_{max}} f(x) \\,dx`
    - Accumulatation of the logistic model over its interquartile range: :math:`A_{iqr} = \\int_{X_{Q1}}^{X_{Q3}} f(x) \\,dx`
    - Average rate of change of the logistic model over its range: :math:`m_{range} = \\frac{f(X_{max}) - f(X_{min})}{X_{max} - X_{min}}`
    - Potential x-values at which the logistic model's instantaneous rate of change equals its average rate of change over its range: :math:`x_{m,range} = \\{ c + \\frac{1}{b}\\cdot{\\ln(2m_{range})} - \\frac{1}{b}\\cdot{\\ln\\left(ab - 2m_{range} - \\sqrt{(2m_{range} - ab)^2 - 4m_{range}^2}\\right)}, \\\\ c + \\frac{1}{b}\\cdot{\\ln(2m_{range})} - \\frac{1}{b}\\cdot{\\ln\\left(ab - 2m_{range} + \\sqrt{(2m_{range} - ab)^2 - 4m_{range}^2}\\right)} \\}`
    - Average value of the logistic model over its range: :math:`v_{range} = \\frac{1}{X_{max} - X_{min}}\\cdot{A_{range}}`
    - Potential x-values at which the logistic model's value equals its average value over its range: :math:`x_{v,range} = \\{ c - \\frac{1}{b}\\cdot{\\ln(\\frac{a}{v_{range}} - 1)} \\}`
    - Average rate of change of the logistic model over its interquartile range: :math:`m_{iqr} = \\frac{f(X_{Q3}) - f(X_{Q1})}{X_{Q3} - X_{Q1}}`
    - Potential x-values at which the logistic model's instantaneous rate of change equals its average rate of change over its interquartile range: :math:`x_{m,iqr} = \\{ c + \\frac{1}{b}\\cdot{\\ln(2m_{iqr})} - \\frac{1}{b}\\cdot{\\ln\\left(ab - 2m_{iqr} - \\sqrt{(2m_{iqr} - ab)^2 - 4m_{iqr}^2}\\right)}, \\\\ c + \\frac{1}{b}\\cdot{\\ln(2m_{iqr})} - \\frac{1}{b}\\cdot{\\ln\\left(ab - 2m_{iqr} + \\sqrt{(2m_{iqr} - ab)^2 - 4m_{iqr}^2}\\right)} \\}`
    - Average value of the logistic model over its interquartile range: :math:`v_{iqr} = \\frac{1}{X_{Q3} - X_{Q1}}\\cdot{A_{iqr}}`
    - Potential x-values at which the logistic model's value equals its average value over its interquartile range: :math:`x_{v,iqr} = \\{ c - \\frac{1}{b}\\cdot{\\ln(\\frac{a}{v_{iqr}} - 1)} \\}`
    - Predicted values based on the logistic model: :math:`\\hat{y}_i = \\{ \\hat{y}_1, \\hat{y}_2, \\cdots, \\hat{y}_n \\}`
    - Residuals of the dependent variable: :math:`e_i = \\{ p_{1,y} - \\hat{y}_1, p_{2,y} - \\hat{y}_2, \\cdots, p_{n,y} - \\hat{y}_n \\}`
    - Deviations of the dependent variable: :math:`d_i = \\{ p_{1,y} - \\bar{y}, p_{2,y} - \\bar{y}, \\cdots, p_{n,y} - \\bar{y} \\}`
    - Sum of squares of residuals: :math:`SS_{res} = \\sum\\limits_{i=1}^n e_i^2`
    - Sum of squares of deviations: :math:`SS_{dev} = \\sum\\limits_{i=1}^n d_i^2`
    - Correlation coefficient for the logistic model: :math:`r = \\sqrt{1 - \\frac{SS_{res}}{SS_{dev}}}`
    - |regression_analysis|

    Examples
    --------
    Import `logistic_model` function from `regressions` library
        >>> from regressions.models.logistic import logistic_model
    Generate a logistic regression model for the data set [[1, 0.0000122], [2, 0.000247], [3, 0.004945], [4, 0.094852], [5, 1.0], [6, 1.905148], [7, 1.995055], [8, 1.999753], [9, 1.999988], [10, 1.999999]], then print its coefficients, roots, total accumulation over its interquartile range, and correlation
        >>> model_perfect = logistic_model([[1, 0.0000122], [2, 0.000247], [3, 0.004945], [4, 0.094852], [5, 1.0], [6, 1.905148], [7, 1.995055], [8, 1.999753], [9, 1.999988], [10, 1.999999]])
        >>> print(model_perfect['constants'])
        [2.0, 3.0, 5.0]
        >>> print(model_perfect['points']['roots'])
        [None]
        >>> print(model_perfect['accumulations']['iqr'])
        5.9987
        >>> print(model_perfect['correlation'])
        1.0
    Generate a logistic regression model for the data set [[1, 32], [2, 25], [3, 14], [4, 23], [5, 39], [6, 45], [7, 42], [8, 49], [9, 36], [10, 33]], then print its coefficients, inflections, total accumulation over its range, and correlation
        >>> model_agnostic = logistic_model([[1, 32], [2, 25], [3, 14], [4, 23], [5, 39], [6, 45], [7, 42], [8, 49], [9, 36], [10, 33]])
        >>> print(model_agnostic['constants'])
        [43.9838, 0.3076, 0.9747]
        >>> print(model_agnostic['points']['inflections'])
        [[0.9747, 21.9919]]
        >>> print(model_agnostic['accumulations']['range'])
        305.9347
        >>> print(model_agnostic['correlation'])
        0.5875
    """
    # Handle input errors
    matrix_of_scalars(data, 'first')
    long_vector(data)
    positive_integer(precision)

    # Store independent and dependent variable values separately
    independent_variable = single_dimension(data, 1)
    dependent_variable = single_dimension(data, 2)

    # Determine key values for bounds
    halved_data = half_dimension(data, 1)
    dependent_lower = single_dimension(halved_data['lower'], 2)
    dependent_upper = single_dimension(halved_data['upper'], 2)
    mean_lower = mean_value(dependent_lower)
    mean_upper = mean_value(dependent_upper)
    dependent_max = max(dependent_variable)
    dependent_min = min(dependent_variable)
    dependent_range = dependent_max - dependent_min
    independent_max = max(independent_variable)
    independent_min = min(independent_variable)
    independent_range = independent_max - independent_min
    independent_avg = (independent_max + independent_min) / 2

    # Circumvent error with bounds
    if dependent_range == 0:
        dependent_range = 1
    if independent_range == 0:
        independent_range = 1
    
    # Create function to guide model generation
    def logistic_fit(variable, first_constant, second_constant, third_constant):
        evaluation = first_constant / (1 + exp(-1 * second_constant * (variable - third_constant)))
        return evaluation
    
    # Create list to store coefficients of generated equation
    solution = []
    
    # Handle normal case where values appear to increase in the set
    if mean_upper >= mean_lower:
        # Generate model
        parameters, covariance = curve_fit(logistic_fit, independent_variable, dependent_variable, bounds=[(dependent_max - dependent_range, 0, independent_avg - independent_range), (dependent_max + dependent_range, inf, independent_avg + independent_range)])
        solution = list(parameters)
    
    # Handle case where values do not appear to increase in the set
    else:
        # Generate model with inverted negative infinity and zero values
        parameters, covariance = curve_fit(logistic_fit, independent_variable, dependent_variable, bounds=[(dependent_max - dependent_range, -inf, independent_avg - independent_range), (dependent_max + dependent_range, 0, independent_avg + independent_range)])
        solution = list(parameters)
    
    # Eliminate zeroes from solution
    coefficients = no_zeroes(solution, precision)

    # Generate evaluations for function, derivative, and integral
    equation = logistic_equation(*coefficients, precision)
    derivative = logistic_derivatives(*coefficients, precision)['first']['evaluation']
    integral = logistic_integral(*coefficients, precision)['evaluation']

    # Determine key points of graph
    points = key_coordinates('logistic', coefficients, precision)

    # Generate values for lower and upper bounds
    five_numbers = five_number_summary(independent_variable, precision)
    min_value = five_numbers['minimum']
    max_value = five_numbers['maximum']
    q1 = five_numbers['q1']
    q3 = five_numbers['q3']

    # Calculate accumulations
    accumulated_range = accumulated_area('logistic', coefficients, min_value, max_value, precision)
    accumulated_iqr = accumulated_area('logistic', coefficients, q1, q3, precision)

    # Determine average values and their points
    averages_range = average_values('logistic', coefficients, min_value, max_value, precision)
    averages_iqr = average_values('logistic', coefficients, q1, q3, precision)

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