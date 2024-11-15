from regressions.errors.vectors import vector_of_scalars
from .sort import sorted_list

def maximum_value(data):
    """
    Determines the largest value of a data set

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional
    TypeError
        Elements of argument must be integers or floats

    Returns
    -------
    maximum : int or float
        Largest value from the data set

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`, :func:`~regressions.statistics.minimum.minimum_value`, :func:`~regressions.statistics.median.median_value`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Maximum value of set: :math:`a_{max} \\geq a_j, \\forall a_j \\in a_i`
    - |maximum|

    Examples
    --------
    Import `maximum_value` function from `regressions` library
        >>> from regressions.statistics.maximum import maximum_value
    Determine the maximum of the set [21, 53, 3, 68, 43, 9, 72, 19, 20, 1]
        >>> maximum_even = maximum_value([21, 53, 3, 68, 43, 9, 72, 19, 20, 1])
        >>> print(maximum_even)
        72
    Determine the maximum of the set [12, 81, 13, 8, 42, 72, 91, 20, 20]
        >>> maximum_odd = maximum_value([12, 81, 13, 8, 42, 72, 91, 20, 20])
        >>> print(maximum_odd)
        91
    """
    # Handle input errors
    vector_of_scalars(data)

    # Sort input
    sorted_data = sorted_list(data)

    # Grab last element of sorted input
    result = sorted_data[-1]
    return result