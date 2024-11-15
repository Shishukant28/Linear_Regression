from .errors.matrices import matrix_of_scalars
from .errors.vectors import long_vector
from .errors.scalars import positive_integer
from .models.linear import linear_model
from .models.quadratic import quadratic_model
from .models.cubic import cubic_model
from .models.hyperbolic import hyperbolic_model
from .models.exponential import exponential_model
from .models.logarithmic import logarithmic_model
from .models.logistic import logistic_model
from .models.sinusoidal import sinusoidal_model
from .vectors.dimension import single_dimension
from .statistics.minimum import minimum_value
from .statistics.maximum import maximum_value
from .statistics.quartiles import quartile_value
from .statistics.mean import mean_value
from .statistics.median import median_value

def run_all(data, precision = 4):
    """
    Generates all eight key regression models (linear, quadratic, cubic, hyperbolic, exponential, logarithmic, logistic, and sinusoidal) for a given data set, in addition to determining the best fit based on correlation and providing various statistical measures

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
    results['models']['linear'] : dict
        See :ref:`Linear Model`
    results['models']['quadratic'] : dict
        See :ref:`Quadratic Model`
    results['models']['cubic'] : dict
        See :ref:`Cubic Model`
    results['models']['hyperbolic'] : dict
        See :ref:`Hyperbolic Model`
    results['models']['exponential'] : dict
        See :ref:`Exponential Model`
    results['models']['logarithmic'] : dict
        See :ref:`Logarithmic Model`
    results['models']['logistic'] : dict
        See :ref:`Logistic Model`
    results['models']['sinusoidal'] : dict
        See :ref:`Sinusoidal Model`
    results['statistics']['minimum'] : int or float
        Smallest value of the independent variable from the provided data set
    results['statistics']['maximum'] : int or float
        Largest value of the independent variable from the provided data set
    results['statistics']['q1'] : int or float
        First quartile of the independent variable from the provided data set, below which 25% of the data points lie
    results['statistics']['q3'] : int or float
        Third quartile of the independent variable from the provided data set, above which 25% of the data points lie
    results['statistics']['mean'] : int or float
        Arithmetic mean of the independent variable from the provided data set
    results['statistics']['median'] : int or float
        Median of the independent variable from the provided data set, which splits the data in half
    results['optimal']['option'] : str
        Name of the model with the highest correlation for this particular data set (e.g., 'cubic' if the cubic model has a higher correlation than all other seven models)
    results['optimal']['correlation'] : float
        Value of the correlation for the model with the best fit (i.e., the model listed in ['optimal']['option'])

    See Also
    --------
    :func:`~regressions.models.linear.linear_model`, :func:`~regressions.models.quadratic.quadratic_model`, :func:`~regressions.models.cubic.cubic_model`, :func:`~regressions.models.hyperbolic.hyperbolic_model`, :func:`~regressions.models.exponential.exponential_model`, :func:`~regressions.models.logarithmic.logarithmic_model`, :func:`~regressions.models.logistic.logistic_model`, :func:`~regressions.models.sinusoidal.sinusoidal_model`, :func:`~regressions.statistics.summary.five_number_summary`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Provided ordered pairs for the data set: :math:`p_i = \\{ (p_{1,x}, p_{1,y}), (p_{2,x}, p_{2,y}), \\cdots, (p_{n,x}, p_{n,y}) \\}`
    - Provided values for the independent variable: :math:`X_i = \\{ p_{1,x}, p_{2,x}, \\cdots, p_{n,x} \\}`
    - Provided values for the dependent variable: :math:`Y_i = \\{ p_{1,y}, p_{2,y}, \\cdots, p_{n,y} \\}`
    - Resultant values for the coefficients of the linear model: :math:`C_{lin} = \\{ a_{lin}, b_{lin} \\}`
    - Standard form for the equation of the linear model: :math:`lin(x) = a_{lin}\\cdot{x} + b_{lin}`
    - Resultant values for the coefficients of the quadratic model: :math:`C_{quad} = \\{ a_{quad}, b_{quad}, c_{quad} \\}`
    - Standard form for the equation of the quadratic model: :math:`quad(x) = a_{quad}\\cdot{x^2} + b_{quad}\\cdot{x} + c_{quad}`
    - Resultant values for the coefficients of the cubic model: :math:`C_{cub} = \\{ a_{cub}, b_{cub}, c_{cub}, d_{cub} \\}`
    - Standard form for the equation of the cubic model: :math:`cub(x) = a_{cub}\\cdot{x^3} + b_{cub}\\cdot{x^2} + c_{cub}\\cdot{x} + d_{cub}`
    - Resultant values for the coefficients of the hyperbolic model: :math:`C_{hyp} = \\{ a_{hyp}, b_{hyp} \\}`
    - Standard form for the equation of the hyperbolic model: :math:`hyp(x) = a_{hyp}\\cdot{\\frac{1}{x}} + b_{hyp}`
    - Resultant values for the coefficients of the exponential model: :math:`C_{exp} = \\{ a_{exp}, b_{exp} \\}`
    - Standard form for the equation of the exponential model: :math:`exp(x) = a_{exp}\\cdot{b_{exp}^x}`
    - Resultant values for the coefficients of the logarithmic model: :math:`C_{log} = \\{ a_{log}, b_{log} \\}`
    - Standard form for the equation of the logarithmic model: :math:`log(x) = a_{log}\\cdot{\\ln{x}} + b_{log}`
    - Resultant values for the coefficients of the logistic model: :math:`C_{lst} = \\{ a_{lst}, b_{lst}, c_{lst} \\}`
    - Standard form for the equation of the logistic model: :math:`lst(x) = \\frac{a_{lst}}{1 + \\text{e}^{-b_{lst}\\cdot(x - c_{lst})}}`
    - Resultant values for the coefficients of the sinusoidal model: :math:`C_{sin} = \\{ a_{sin}, b_{sin}, c_{sin}, d_{sin} \\}`
    - Standard form for the equation of the sinusoidal model: :math:`sin(x) = a_{sin}\\cdot{\\sin(b_{sin}\\cdot(x - c_{sin}))} + d_{sin}`
    - |regression_analysis|

    Examples
    --------
    Import `run_all` function from `regressions` library
        >>> from regressions.execute import run_all
    Generate all eight regression models for the data set [[1, 32], [2, 25], [3, 14], [4, 23], [5, 39], [6, 45], [7, 42], [8, 49], [9, 36], [10, 33]], then print each model's coefficients, the mean of the data set, and the name of the model with the best fit
        >>> results = run_all([[1, 32], [2, 25], [3, 14], [4, 23], [5, 39], [6, 45], [7, 42], [8, 49], [9, 36], [10, 33]])
        >>> print(results['models']['linear']['constants'])
        [1.9636, 23.0]
        >>> print(results['models']['quadratic']['constants'])
        [-0.3106, 5.3803, 16.1667]
        >>> print(results['models']['cubic']['constants'])
        [-0.3881, 6.0932, -24.155, 49.4667]
        >>> print(results['models']['hyperbolic']['constants'])
        [-13.5246, 37.7613]
        >>> print(results['models']['exponential']['constants'])
        [22.1049, 1.0692]
        >>> print(results['models']['logarithmic']['constants'])
        [7.4791, 22.5032]
        >>> print(results['models']['logistic']['constants'])
        [43.9838, 0.3076, 0.9747]
        >>> print(results['models']['sinusoidal']['constants'])
        [14.0875, 0.7119, -3.7531, 34.2915]
        >>> print(results['statistics']['mean'])
        5.5
        >>> print(results['optimal']['option'])
        'sinusoidal'
    Generate all eight regression models for the data set [[169, 423], [122, 391], [178, 555], [131, 284], [120, 520], [179, 558], [164, 265], [167, 338], [198, 445], [139, 402], [183, 725], [133, 470], [156, 573], [159, 325], [121, 653], [118, 358], [122, 633], [167, 487], [161, 453], [194, 488], [170, 517], [124, 377], [191, 310], [194, 398], [173, 744], [166, 389], [113, 583], [109, 380], [126, 668], [144, 491], [107, 533], [188, 355], [147, 553], [169, 497], [121, 606], [132, 373], [111, 554], [173, 669], [177, 483], [122, 340], [171, 286], [108, 681], [139, 502], [115, 339], [174, 396], [134, 625], [147, 435], [146, 555], [147, 656], [126, 354], [155, 679], [181, 629], [149, 417], [119, 374], [102, 422], [112, 292], [108, 464], [109, 559], [112, 635], [159, 518], [180, 304], [185, 567], [165, 299], [160, 337], [133, 730], [193, 374], [164, 537], [172, 592], [173, 660], [186, 290], [170, 670], [192, 687], [154, 596], [154, 464], [125, 383], [193, 559], [155, 586], [149, 406], [131, 590], [127, 339], [163, 378], [145, 254], [156, 395], [166, 355], [189, 661], [133, 685], [168, 685], [190, 736], [145, 564], [125, 470], [129, 541], [133, 439], [162, 486], [125, 387], [183, 596], [135, 733], [106, 329], [100, 279], [102, 439], [162, 454]], then print each model's coefficients, the mean of the data set, and the name of the model with the best fit
        >>> results_large = run_all([[169, 423], [122, 391], [178, 555], [131, 284], [120, 520], [179, 558], [164, 265], [167, 338], [198, 445], [139, 402], [183, 725], [133, 470], [156, 573], [159, 325], [121, 653], [118, 358], [122, 633], [167, 487], [161, 453], [194, 488], [170, 517], [124, 377], [191, 310], [194, 398], [173, 744], [166, 389], [113, 583], [109, 380], [126, 668], [144, 491], [107, 533], [188, 355], [147, 553], [169, 497], [121, 606], [132, 373], [111, 554], [173, 669], [177, 483], [122, 340], [171, 286], [108, 681], [139, 502], [115, 339], [174, 396], [134, 625], [147, 435], [146, 555], [147, 656], [126, 354], [155, 679], [181, 629], [149, 417], [119, 374], [102, 422], [112, 292], [108, 464], [109, 559], [112, 635], [159, 518], [180, 304], [185, 567], [165, 299], [160, 337], [133, 730], [193, 374], [164, 537], [172, 592], [173, 660], [186, 290], [170, 670], [192, 687], [154, 596], [154, 464], [125, 383], [193, 559], [155, 586], [149, 406], [131, 590], [127, 339], [163, 378], [145, 254], [156, 395], [166, 355], [189, 661], [133, 685], [168, 685], [190, 736], [145, 564], [125, 470], [129, 541], [133, 439], [162, 486], [125, 387], [183, 596], [135, 733], [106, 329], [100, 279], [102, 439], [162, 454]])
        >>> print(results_large['models']['linear']['constants'])
        [0.4934, 414.5401]
        >>> print(results_large['models']['quadratic']['constants'])
        [-0.007, 2.5668, 265.4919]
        >>> print(results_large['models']['cubic']['constants'])
        [0.0005, -0.2204, 33.8099, -1226.1398]
        >>> print(results_large['models']['hyperbolic']['constants'])
        [-10786.2465, 563.019]
        >>> print(results_large['models']['exponential']['constants'])
        [407.8094, 1.0009]
        >>> print(results_large['models']['logarithmic']['constants'])
        [74.0076, 118.997]
        >>> print(results_large['models']['logistic']['constants'])
        [878.3475, 0.0023, 51.0002]
        >>> print(results_large['models']['sinusoidal']['constants'])
        [32.3199, 1.0085, 1.8848, 488.9635]
        >>> print(results_large['statistics']['mean'])
        149.29
        >>> print(results_large['optimal']['option'])
        'sinusoidal'
    """
    # Handle input errors
    matrix_of_scalars(data, 'first')
    long_vector(data)
    positive_integer(precision)

    # Grab values of independent variable
    independent_variable = single_dimension(data, 1)

    # Generate all eight key regression models
    models = {
        'linear': linear_model(data, precision),
        'quadratic': quadratic_model(data, precision),
        'cubic': cubic_model(data, precision),
        'hyperbolic': hyperbolic_model(data, precision),
        'exponential': exponential_model(data, precision),
        'logarithmic': logarithmic_model(data, precision),
        'logistic': logistic_model(data, precision),
        'sinusoidal': sinusoidal_model(data, precision)
    }

    # Determine key statistical values for independent variable
    statistics = {
        'minimum': minimum_value(independent_variable),
        'maximum': maximum_value(independent_variable),
        'q1': quartile_value(independent_variable, 1),
        'q3': quartile_value(independent_variable, 3),
        'mean': mean_value(independent_variable),
        'median': median_value(independent_variable)
    }

    # Grab correlations of all previously generated models
    correlations = {
        'linear': models['linear']['correlation'],
        'quadratic': models['quadratic']['correlation'],
        'cubic': models['cubic']['correlation'],
        'hyperbolic': models['hyperbolic']['correlation'],
        'exponential': models['exponential']['correlation'],
        'logarithmic': models['logarithmic']['correlation'],
        'logistic': models['logistic']['correlation'],
        'sinusoidal': models['sinusoidal']['correlation']
    }

    # Determine model with highest correlation
    best = max(correlations, key=correlations.get)
    optimal = {
        'option': best,
        'correlation': correlations[best]
    }

    # Package preceding results in single dictionary to return
    result = {
        'models': models,
        'statistics': statistics,
        'optimal': optimal
    }
    return result