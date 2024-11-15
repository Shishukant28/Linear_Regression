from regressions.errors.vectors import vector_of_scalars

def sum_value(data):
    """
    Calculates the sum of all elements in a data set

    Parameters
    ----------
    data : list of int or float
        List of numbers

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional list
    TypeError
        Elements of argument must be integers or floats

    Returns
    -------
    total : float
        Number representing the sum of all elements in the original set

    See Also
    --------
    :func:`~regressions.statistics.mean.mean_value`, :func:`~regressions.statistics.correlation.correlation_coefficient`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Sum of all numbers in set: :math:`\\sum\\limits_{i=1}^n a_i = a_1 + a_2 + \\cdots + a_n`
    - |summation_notation|

    Examples
    --------
    Import `sum_value` function from `regressions` library
        >>> from regressions.statistics.summation import sum_value
    Find the total sum of all values in the array [2, 3, 5, 7]
        >>> total_1 = sum_value([2, 3, 5, 7])
        >>> print(total_1)
        17.0
    Find the total sum of all values in the array [1, -1, 1, -1]
        >>> total_2 = sum_value([1, -1, 1, -1])
        >>> print(total_2)
        0.0
    """
    # Handle input errors
    vector_of_scalars(data)

    # Create number to return
    result = 0

    # Iterate over input
    for element in data:
        # Add each element from input to number to return
        result += element
    
    # Convert number to float
    floated_result = float(result)
    return floated_result