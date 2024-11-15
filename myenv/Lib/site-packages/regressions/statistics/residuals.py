from regressions.errors.scalars import scalar_value
from regressions.errors.vectors import compare_vectors

def single_residual(actual, expected):
    """
    Calculates the difference between the actual value and the expected value

    Parameters
    ----------
    actual : int or float
        Value actually provided by an initial data set
    expected : int or float
        Value predicted to occur by a generated model at the same input to the one that coincided with the actual value

    Raises
    ------
    TypeError
        Arguments must be integers or floats

    Returns
    -------
    residual : float
        Difference between the actual value and the expected value

    See Also
    --------
    :func:`~regressions.statistics.deviations.single_deviation`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Observed value: :math:`y`
    - Predicted value: :math:`\\hat{y}`
    - Residual: :math:`e = y - \\hat{y}`
    - |residual|

    Examples
    --------
    Import `single_residual` function from `regressions` library
        >>> from regressions.statistics.residuals import single_residual
    Determine the residual between an actual value of 7.8 and an expected value of 9.2
        >>> residual_small = single_residual(7.8, 9.2)
        >>> print(residual_small)
        -1.3999999999999995
    Determine the residual between an actual value of 6.1 and an expected value of 19.8
        >>> residual_large = single_residual(6.1, 19.8)
        >>> print(residual_large)
        -13.700000000000001
    """
    # Handle input errors
    scalar_value(actual, 'first')
    scalar_value(expected, 'second')

    # Calculate difference between inputs
    difference = actual - expected

    # Convert difference to float
    result = float(difference)
    return result

def multiple_residuals(actual_array, expected_array):
    """
    Generates a list of the differences between the actual values from one list and the expected values from another list

    Parameters
    ----------
    actual_array : list of int or float
        List containing the actual values observed from a data set
    expected_array : list of int or float
        List containing the expected values predicted for a data set

    Raises
    ------
    TypeError
        Arguments must be 1-dimensional lists
    TypeError
        Elements of arguments must be integers or floats
    ValueError
        Both arguments must contain the same number of elements

    Returns
    -------
    residuals : list of float
        Differences between the actual values and the expected values

    See Also
    --------
    :func:`~regressions.statistics.deviations.multiple_deviations`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Observed values: :math:`y_i = \\{ y_1, y_2, \\cdots, y_n \\}`
    - Predicted values: :math:`\\hat{y}_i = \\{ \\hat{y}_1, \\hat{y}_2, \\cdots, \\hat{y}_n \\}`
    - Residuals: :math:`e_i = \\{ y_1 - \\hat{y}_1, y_2 - \\hat{y}_2, \\cdots, y_n - \\hat{y}_n \\}`
    - |residual|

    Examples
    --------
    Import `multiple_residuals` function from `regressions` library
        >>> from regressions.statistics.residuals import multiple_residuals
    Determine the residuals between the actual values [5.6, 8.1, 6.3] and the expected values [6.03, 8.92, 6.12]
        >>> residuals_short = multiple_residuals([5.6, 8.1, 6.3], [6.03, 8.92, 6.12])
        >>> print(residuals_short)
        [-0.4300000000000006, -0.8200000000000003, 0.17999999999999972]
    Determine the residuals between the actual values [11.7, 5.6, 8.1, 13.4, 6.3] and the expected values [15.17, 6.03, 8.92, 9.42, 6.12]
        >>> residuals_long = multiple_residuals([11.7, 5.6, 8.1, 13.4, 6.3], [15.17, 6.03, 8.92, 9.42, 6.12])
        >>> print(residuals_long)
        [-3.4700000000000006, -0.4300000000000006, -0.8200000000000003, 3.9800000000000004, 0.17999999999999972]
    """
    # Handle input errors
    compare_vectors(actual_array, expected_array)

    # Create list to return
    results = []

    # Iterate over inputs
    for i in range(len(actual_array)):
        # Store residuals of corresponding elements in list to return
        results.append(single_residual(actual_array[i], expected_array[i]))
    
    # Return results
    return results