from regressions.errors.vectors import vector_of_strings

def sorted_list(data):
    """
    Sorts all elements in a data set in increasing order via quicksort

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze

    Returns
    -------
    order : list of int or float
        List of all elements from a data set sorted in increasing order

    See Also
    --------
    :func:`~regressions.statistics.minimum.minimum_value`, :func:`~regressions.statistics.maximum.maximum_value`, :func:`~regressions.statistics.median.median_value`, :func:`~regressions.statistics.halve.half`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Sorted version of set: :math:`A_i = ( A_1, A_2, \\cdots, A_n )`

        - For all terms in :math:`A_i`: :math:`A_{n-1} \\leq A_n`

    - |monotonic_function|

    Examples
    --------
    Import `sorted_list` function from `regressions` library
        >>> from regressions.statistics.sort import sorted_list
    Sort the set [5, 2, 9, 8]
        >>> order_1 = sorted_list([5, 2, 9, 8])
        >>> print(order_1)
        [2, 5, 8, 9]
    Sort the set [11, 3, 52, 25, 21, 25, 6]
        >>> order_2 = sorted_list([11, 3, 52, 25, 21, 25, 6])
        >>> print(order_2)
        [3, 6, 11, 21, 25, 25, 52]
    """
    # Create intermediary lists
    pivots = []
    less = []
    more = []

    # Handle base case
    if len(data) <= 1:
        return data
    
    # Handle recursive case
    else:
        # Set pivot
        pivot = data[0]

        # Iterate over input
        for i in data:
            # Store values below pivot in less list
            if i < pivot:
                less.append(i)
            
            # Store values above pivot in more list
            elif i > pivot:
                more.append(i)
            
            # Store values equal to pivot in pivots list
            else:
                pivots.append(i)
        
        # Sort intermediary lists
        less = sorted_list(less)
        more = sorted_list(more)

        # Create final list to return
        result = less + pivots + more
        return result

def sorted_dimension(data, dimension = 1):
    """
    Sorts all elements in a multidimensional data set in increasing order, according to particular dimension

    Parameters
    ----------
    data : list of lists of int or float
        List of lists of numbers to analyze
    dimension: int, default=1
        Number representing the dimension to use for sorting

    Returns
    -------
    order : list of lists of int or float
        List of lists of numbers sorted in increasing order, based on only one dimension of the nested list

    See Also
    --------
    :func:`~regressions.statistics.halve.half_dimension`, :func:`~regressions.vectors.dimension.single_dimension`

    Notes
    -----
    - Set of ordered pairs of numbers: :math:`a_i = \\{ ( a_{1,1}, a_{1,2}, \\cdots, a_{1,j}, a_{1,n} ), ( a_{2,1}, a_{2,2}, \\cdots, a_{2,j}, a_{2,n} ), \\cdots, \\\\ ( a_{m,1}, a_{m,2}, \\cdots, a_{m,j}, a_{m,n} ) \\}`
    - Sorted version of set according to the values in the :math:`j`\ th position: :math:`A_i = ( ( A_{1,1}, A_{1,2}, \\cdots, A_{1,j}, A_{1,n} ), ( A_{2,1}, A_{2,2}, \\cdots, A_{2,j}, A_{2,n} ), \\cdots, \\\\ ( A_{m,1}, A_{m,2}, \\cdots, A_{m,j}, A_{m,n} ) )`

        - For all terms in :math:`A_i`: :math:`A_{n-1,j} \\leq A_{n,j}`

    - |monotonic_function|

    Examples
    --------
    Import `sorted_dimension` function from `regressions` library
        >>> from regressions.statistics.sort import sorted_dimension
    Sort the set [[1, 3, 5], [9, 2, 4], [6, 1, 8]] according to its second dimension
        >>> order_1 = sorted_dimension([[1, 3, 5], [9, 2, 4], [6, 1, 8]], 2)
        >>> print(order_1)
        [[6, 1, 8], [9, 2, 4], [1, 3, 5]]
    Sort the set [[1, 3, 5], [9, 2, 4], [6, 1, 8]] according to its third dimension
        >>> order_2 = sorted_dimension([[1, 3, 5], [9, 2, 4], [6, 1, 8]], 3)
        >>> print(order_2)
        [[9, 2, 4], [1, 3, 5], [6, 1, 8]]
    """
    # Create intermediary lists
    pivots = []
    less = []
    more = []

    # Handle base case
    if len(data) <= 1:
        return data
    
    # Handle recursive case
    else:
        # Set pivot
        pivot = data[0][dimension - 1]

        # Iterate over list
        for i in data:
            # Store values below pivot in less list
            if i[dimension - 1] < pivot:
                less.append(i)
            
            # Store values above pivot in more list
            elif i[dimension - 1] > pivot:
                more.append(i)
            
            # Store values equal to pivot in pivots list
            else:
                pivots.append(i)
        
        # Sort intermediary lists
        less = sorted_dimension(less, dimension)
        more = sorted_dimension(more, dimension)

        # Create final list to return
        result = less + pivots + more
        return result

def sorted_strings(data):
    # Handle input errors
    vector_of_strings(data)

    # Create list to return
    sorted_data = []

    # Handle single entry requiring no sorting
    if len(data) == 1:
        sorted_data = data
    
    # Handle general case of two entries
    else:
        # Determine values of strings
        first_index = data[0].find(' + ')
        first_value = float(data[0][:first_index])
        second_index = data[1].find(' + ')
        second_value = float(data[1][:second_index])
        
        # Set return list to initial list if already sorted
        if first_value < second_value:
            sorted_data = data
        
        # Set return list to inversion of initial list if not already sorted
        else:
            sorted_data = [data[1], data[0]]
    
    # Return result
    return sorted_data