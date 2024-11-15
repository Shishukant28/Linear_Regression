from regressions.errors.vectors import vector_of_scalars
from .summation import sum_value

def mean_value(data):
    """
    Determines the arithmetic mean of a data set

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional list
    TypeError
        Elements of argument must be integers or floats

    Returns
    -------
    mean : float
        Arithmetic mean of the data set

    See Also
    --------
    :func:`~regressions.statistics.summation.sum_value`, :func:`~regressions.statistics.median.median_value`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Mean of all numbers in set: :math:`\\bar{a} = \\frac{1}{n}\\cdot{\\sum\\limits_{i=1}^n a_i}`
    - |mean|

    Examples
    --------
    Import `mean_value` function from `regressions` library
        >>> from regressions.statistics.mean import mean_value
    Determine the mean of the set [21, 53, 3, 68, 43, 9, 72, 19, 20, 1]
        >>> mean_even = mean_value([21, 53, 3, 68, 43, 9, 72, 19, 20, 1])
        >>> print(mean_even)
        30.9
    Determine the mean of the set [12, 81, 13, 8, 42, 72, 91, 20, 20]
        >>> mean_odd = mean_value([12, 81, 13, 8, 42, 72, 91, 20, 20])
        >>> print(mean_odd)
        39.888888888888886
    """
    # Handle input errors
    vector_of_scalars(data)

    # Calculate sum of all elements in input
    sum_of_data = sum_value(data)

    # Determine length of input
    length_of_data = len(data)

    # Calculate average
    average = sum_of_data / length_of_data

    # Convert average to float
    result = float(average)
    return result