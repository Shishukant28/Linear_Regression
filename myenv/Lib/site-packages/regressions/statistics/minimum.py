from regressions.errors.vectors import vector_of_scalars
from .sort import sorted_list

def minimum_value(data):
    """
    Determines the smallest value of a data set

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
    minimum : int or float
        Smallest value from the data set

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`, :func:`~regressions.statistics.maximum.maximum_value`, :func:`~regressions.statistics.median.median_value`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Minimum value of set: :math:`a_{min} \\leq a_j, \\forall a_j \\in a_i`
    - |minimum|

    Examples
    --------
    Import `minimum_value` function from `regressions` library
        >>> from regressions.statistics.minimum import minimum_value
    Determine the minimum of the set [21, 53, 3, 68, 43, 9, 72, 19, 20, 1]
        >>> minimum_even = minimum_value([21, 53, 3, 68, 43, 9, 72, 19, 20, 1])
        >>> print(minimum_even)
        1
    Determine the minimum of the set [12, 81, 13, 8, 42, 72, 91, 20, 20]
        >>> minimum_odd = minimum_value([12, 81, 13, 8, 42, 72, 91, 20, 20])
        >>> print(minimum_odd)
        8
    """
    # Handle input errors
    vector_of_scalars(data)

    # Sort input
    sorted_data = sorted_list(data)

    # Grab first element of sorted input
    result = sorted_data[0]
    return result