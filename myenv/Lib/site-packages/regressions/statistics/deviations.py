from regressions.errors.scalars import scalar_value
from regressions.errors.vectors import vector_of_scalars
from .mean import mean_value

def single_deviation(actual, mean): 
    """
    Calculates the difference between the actual value from a data set and the mean of that data set

    Parameters
    ----------
    actual : int or float
        Value actually provided by an initial data set
    mean : int or float
        Average value of the data set

    Raises
    ------
    TypeError
        Arguments must be integers or floats

    Returns
    -------
    deviation : float
        Difference between the actual value and the mean of the data set

    See Also
    --------
    :func:`~regressions.statistics.mean.mean_value`, :func:`~regressions.statistics.residuals.single_residual`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Observed value: :math:`y`
    - Mean of all observed values: :math:`\\bar{y}`
    - Deviation: :math:`d = y - \\bar{y}`
    - |deviation|

    Examples
    --------
    Import `single_deviation` function from `regressions` library
        >>> from regressions.statistics.deviations import single_deviation
    Determine the deviation for an actual value of 7.8 and a mean of 13.75
        >>> deviation_small = single_deviation(7.8, 13.75)
        >>> print(deviation_small)
        -5.95
    Determine the deviation between an actual value of 6.1 and an mean of -19.7
        >>> deviation_large = single_deviation(6.1, -19.7)
        >>> print(deviation_large)
        25.799999999999997
    """
    # Handle input errors
    scalar_value(actual, 'first')
    scalar_value(mean, 'second')

    # Calcuate difference between actual and mean
    difference = actual - mean

    # Convert difference to float
    result = float(difference)
    return result

def multiple_deviations(actual_array):
    """
    Generates a list of the differences between the actual values from an original list and mean value of the actual values from that original list

    Parameters
    ----------
    actual_array : list of int or float
        List containing the actual values observed from a data set

    Raises
    ------
    TypeError
        Arguments must be 1-dimensional lists
    TypeError
        Elements of arguments must be integers or floats

    Returns
    -------
    deviations : list of float
        List of differences between the actual values and the mean value for all elements from the original list

    See Also
    --------
    :func:`~regressions.statistics.mean.mean_value`, :func:`~regressions.statistics.residuals.multiple_residuals`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Observed values: :math:`y_i = \\{ y_1, y_2, \\cdots, y_n \\}`
    - Mean of all observed values: :math:`\\bar{y} = \\frac{1}{n}\\cdot{\\sum\\limits_{i=1}^n y_i}`
    - Deviations: :math:`d_i = \\{ y_1 - \\bar{y}, y_2 - \\bar{y}, \\cdots, y_n - \\bar{y} \\}`
    - |deviation|

    Examples
    --------
    Import `multiple_deviations` function from `regressions` library
        >>> from regressions.statistics.deviations import multiple_deviations
    Generate a list of deviations from this data set [8.2, 9.41, 1.23, 34.7]
        >>> deviations_short = multiple_deviations([8.2, 9.41, 1.23, 34.7])
        >>> print(deviations_short)
        [-5.185000000000002, -3.9750000000000014, -12.155000000000001, 21.315]
    Generate a list of deviations from this data set [5.21, 8.2, 9.41, 1.23, 10.52, 21.76, 34.7]
        >>> deviations_long = multiple_deviations([5.21, 8.2, 9.41, 1.23, 10.52, 21.76, 34.7])
        >>> print(deviations_long)
        [-7.7942857142857145, -4.804285714285715, -3.5942857142857143, -11.774285714285714, -2.484285714285715, 8.755714285714287, 21.69571428571429]
    """
    # Handle input errors
    vector_of_scalars(actual_array)

    # Create list to return
    results = []

    # Calculate mean of input
    average = mean_value(actual_array)

    # Iterate over input
    for element in actual_array:
        # Store deviation of each element in list to return
        results.append(single_deviation(element, average))
    
    # Return results
    return results