from regressions.errors.vectors import compare_vectors
from regressions.errors.scalars import positive_integer
from .residuals import multiple_residuals
from .deviations import multiple_deviations
from .summation import sum_value
from .rounding import rounded_value

def correlation_coefficient(actuals, expecteds, precision = 4):
    """
    Calculates the correlation coefficient as a way to predict the strength of a predicted model by comparing the ratio of residuals to deviations, in order to determine a strong or weak relationship

    Parameters
    ----------
    actuals : list of int or float
        List containing the actual values observed from a data set
    expecteds : list of int or float
        List containing the expected values for a data set based on a predictive model
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the result

    Raises
    ------
    TypeError
        First and second arguments must be 1-dimensional lists
    TypeError
        Elements of first and second arguments must be integers or floats
    ValueError
        First and second arguments must contain the same number of elements
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    correlation : float
        Number indicating statistical strenght of the relationship between two variables; the closer to 1, the stronger; the closer to 0, the weaker

    See Also
    --------
    :func:`~regressions.statistics.residuals.multiple_residuals`, :func:`~regressions.statistics.deviations.multiple_deviations`, :func:`~regressions.statistics.summation.sum_value`

    Notes
    -----
    - Observed values: :math:`y_i = \\{ y_1, y_2, \\cdots, y_n \\}`
    - Predicted values: :math:`\\hat{y}_i = \\{ \\hat{y}_1, \\hat{y}_2, \\cdots, \\hat{y}_n \\}`
    - Mean of all observed values: :math:`\\bar{y} = \\frac{1}{n}\\cdot{\\sum\\limits_{i=1}^n y_i}`
    - Residuals: :math:`e_i = \\{ y_1 - \\hat{y}_1, y_2 - \\hat{y}_2, \\cdots, y_n - \\hat{y}_n \\}`
    - Deviations: :math:`d_i = \\{ y_1 - \\bar{y}, y_2 - \\bar{y}, \\cdots, y_n - \\bar{y} \\}`
    - Sum of squares of residuals: :math:`SS_{res} = \\sum\\limits_{i=1}^n e_i^2`
    - Sum of squares of deviations: :math:`SS_{dev} = \\sum\\limits_{i=1}^n d_i^2`
    - Correlation coefficient: :math:`r = \\sqrt{1 - \\frac{SS_{res}}{SS_{dev}}}`
    - |determination|

    Examples
    --------
    Import `correlation_coefficient` function from `regressions` library
        >>> from regressions.statistics.correlation import correlation_coefficient
    Calculate the correlation using the provided actual values [8.2, 9.41, 1.23, 34.7] and the predicted values [7.863, 8.9173, 2.0114, 35.8021]
        >>> correlation_short = correlation_coefficient([8.2, 9.41, 1.23, 34.7], [7.863, 8.9173, 2.0114, 35.8021])
        >>> print(correlation_short)
        0.9983
    Calculate the correlation using the provided actual values [2, 3, 5, 7, 11, 13, 17, 19] and the predicted values [1.0245, 3.7157, 6.1398, 8.1199, 12.7518, 14.9621, 15.2912, 25.3182]
        >>> correlation_long = correlation_coefficient([2, 3, 5, 7, 11, 13, 17, 19], [1.0245, 3.7157, 6.1398, 8.1199, 12.7518, 14.9621, 15.2912, 25.3182])
        >>> print(correlation_long)
        0.9011
    """
    # Handle input errors
    compare_vectors(actuals, expecteds)
    positive_integer(precision)

    # Determine residuals and deviations between actuals and expecteds
    residuals = multiple_residuals(actuals, expecteds)
    deviations = multiple_deviations(actuals)

    # Square residuals and deviations
    squared_residuals = []
    for residual in residuals:
        squared_residuals.append(residual**2)
    squared_deviations = []
    for deviation in deviations:
        squared_deviations.append(deviation**2)
    
    # Calculate the sums of the squares of the residuals and deviations
    residual_sum = sum_value(squared_residuals)
    deviation_sum = sum_value(squared_deviations)

    # Circumvent division by zero
    if deviation_sum == 0:
        deviation_sum = 10**(-precision)
    
    # Calculate key ratio
    ratio = residual_sum / deviation_sum
    
    # Handle no solution
    if ratio > 1:
        return 0.0
    
    # Handle general case
    else:
        result = (1 - ratio)**(1/2)
        return rounded_value(result, precision)