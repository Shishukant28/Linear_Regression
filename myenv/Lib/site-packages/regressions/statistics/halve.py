from math import floor
from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import positive_integer
from regressions.errors.matrices import matrix_of_scalars
from .sort import sorted_list, sorted_dimension

def partition(data):
    """
    Splits an unsorted data set into two unsorted data sets, each containing the same amount of elements, with elements allocated soley according to where they happen to appear in the original data set (in sets with an odd amount of elements, the median is not included in either half)

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze

    Returns
    -------
    sections['upper'] : list of int or float
        List of all elements from the upper half of a data set
    sections['lower'] : list of int or float
        List of all elements from the lower half of a data set

    See Also
    --------
    :func:`~regressions.vectors.dimension.single_dimension`, :func:`~regressions.vectors.unify.unite_vectors`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - For sets with an odd amount of numbers:

        - Lower section: :math:`a_{lower} = \\{ a_1, a_2, \\cdots, a_{\\lfloor n/2 \\rfloor} \\}`
        - Upper section: :math:`a_{upper} = \\{ a_{\\lceil n/2 \\rceil}, a_{\\lceil n/2 \\rceil + 1}, \\cdots, a_n \\}`
    
    - For sets with an even amount of numbers:

        - Lower section: :math:`a_{lower} = \\{ a_1, a_2, \\cdots, a_{n/2} \\}`
        - Upper section: :math:`a_{upper} = \\{ a_{n/2 + 1}, a_{n/2 + 2}, \\cdots, a_n \\}`

    Examples
    --------
    Import `partition` function from `regressions` library
        >>> from regressions.statistics.halve import partition
    Determine the upper half of the set [5, 2, 9, 8]
        >>> sections_short = partition([5, 2, 9, 8])
        >>> print(sections_short['upper'])
        [9, 8]
    Determine the lower half of the set [11, 3, 52, 25, 21, 25, 6]
        >>> sections_long = partition([11, 3, 52, 25, 21, 25, 6])
        >>> print(sections_long['lower'])
        [11, 3, 52]
    """
    # Determine length of input
    length = len(data)

    # Create intermediary lists
    upper = []
    lower = []

    # Handle an even amount of elements
    if length % 2 == 0:
        index = int(length / 2)
        upper = data[index:]
        lower = data[:index]
    
    # Handle an odd amount of elements
    else:
        index = int(floor(length / 2))
        upper = data[index + 1:]
        lower = data[:index]
    
    # Package both lists in single dictionary to return
    result = {
        'upper': upper,
        'lower': lower
    }
    return result

def half(data):
    """
    Splits an unsorted data set into two sorted data sets, each containing the same amount of elements (in sets with an odd amount of elements, the median is not included in either half)

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
    sections['upper'] : list of int or float
        List of all elements from the upper half of a sorted data set
    sections['lower'] : list of int or float
        List of all elements from the lower half of a sorted data set

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Sorted version of set: :math:`A_i = ( A_1, A_2, \\cdots, A_n )`

        - For all terms in :math:`A_i`: :math:`A_{n-1} \\leq A_n`

    - For sets with an odd amount of numbers:

        - Lower section: :math:`A_{lower} = ( A_1, A_2, \\cdots, A_{\\lfloor n/2 \\rfloor} )`
        - Upper section: :math:`A_{upper} = ( A_{\\lceil n/2 \\rceil}, A_{\\lceil n/2 \\rceil + 1}, \\cdots, A_n )`
    
    - For sets with an even amount of numbers:

        - Lower section: :math:`A_{lower} = ( A_1, A_2, \\cdots, A_{n/2} )`
        - Upper section: :math:`A_{upper} = ( A_{n/2 + 1}, A_{n/2 + 2}, \\cdots, A_n )`

    Examples
    --------
    Import `half` function from `regressions` library
        >>> from regressions.statistics.halve import half
    Determine the sorted upper half of the set [5, 2, 9, 8]
        >>> sections_short = half([5, 2, 9, 8])
        >>> print(sections_short['upper'])
        [8, 9]
    Determine the sorted lower half of the set [11, 3, 52, 25, 21, 25, 6]
        >>> sections_long = half([11, 3, 52, 25, 21, 25, 6])
        >>> print(sections_long['lower'])
        [3, 6, 11]
    """
    # Handle input errors
    vector_of_scalars(data)

    # Sort input
    sorted_data = sorted_list(data)

    # Partition sorted data
    result = partition(sorted_data)
    return result

def half_dimension(data, dimension = 1):
    """
    Splits an unsorted 2-dimensional data set into two sorted 2-dimensional data sets, each containing the same amount of elements, in which the sorting occurs based on the elements of the nested lists indicated by the dimension parameter (in sets with an odd amount of elements, the median is not included in either half)

    Parameters
    ----------
    data : list of lists of int or float
        List of lists of numbers to analyze
    dimension : int, default=1
        Number indicating by which element of the nested lists to sort
    
    Raises
    ------
    TypeError
        First argument must be a 2-dimensional list
    TypeError
        Elements nested within first argument must be integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    sections['upper'] : list of lists of int or float
        List of all elements from the upper half of a data set, sorted according to the elements occupying a provided position
    sections['lower'] : list of lists of int or float
        List of all elements from the lower half of a data set, sorted according to the elements occupying a provided position

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_dimension`, :func:`~regressions.vectors.dimension.single_dimension`

    Notes
    -----
    - Set of ordered pairs of numbers: :math:`a_i = \\{ ( a_{1,1}, a_{1,2}, \\cdots, a_{1,j}, a_{1,n} ), ( a_{2,1}, a_{2,2}, \\cdots, a_{2,j}, a_{2,n} ), \\cdots, \\\\ ( a_{m,1}, a_{m,2}, \\cdots, a_{m,j}, a_{m,n} ) \\}`
    - Sorted version of set according to the values in the :math:`j`\ th position: :math:`A_i = ( ( A_{1,1}, A_{1,2}, \\cdots, A_{1,j}, A_{1,n} ), ( A_{2,1}, A_{2,2}, \\cdots, A_{2,j}, A_{2,n} ), \\cdots, \\\\ ( A_{m,1}, A_{m,2}, \\cdots, A_{m,j}, A_{m,n} ) )`

        - For all terms in :math:`A_i`: :math:`A_{n-1,j} \\leq A_{n,j}`
    
    - For sets with an odd amount of ordered pairs:

        - Lower section: :math:`A_{lower} = ( ( A_{1,1}, A_{1,2}, \\cdots, A_{1,j}, A_{1,n} ), ( A_{2,1}, A_{2,2}, \\cdots, A_{2,j}, A_{2,n} ), \\cdots, \\\\ ( A_{\\lfloor m/2 \\rfloor,1}, A_{\\lfloor m/2 \\rfloor,2}, \\cdots, A_{\\lfloor m/2 \\rfloor,j}, A_{\\lfloor m/2 \\rfloor,n} ) )`
        - Upper section: :math:`A_{upper} = ( ( A_{\\lceil m/2 \\rceil,1}, A_{\\lceil m/2 \\rceil,2}, \\cdots, A_{\\lceil m/2 \\rceil,j}, A_{\\lceil m/2 \\rceil,n} ), ( A_{\\lceil m/2 \\rceil + 1,1}, A_{\\lceil m/2 \\rceil + 1,2}, \\cdots, \\\\ A_{\\lceil m/2 \\rceil + 1,j}, A_{\\lceil m/2 \\rceil + 1,n} ), \\cdots, ( A_{m,1}, A_{m,2}, \\cdots, A_{m,j}, A_{m,n} ) )`
    
    - For sets with an even amount of ordered pairs:

        - Lower section: :math:`A_{lower} = ( ( A_{1,1}, A_{1,2}, \\cdots, A_{1,j}, A_{1,n} ), ( A_{2,1}, A_{2,2}, \\cdots, A_{2,j}, A_{2,n} ), \\cdots, \\\\ ( A_{m/2,1}, A_{m/2,2}, \\cdots, A_{m/2,j}, A_{m/2,n} ) )`
        - Upper section: :math:`A_{upper} = ( ( A_{m/2 + 1,1}, A_{m/2 + 1,2}, \\cdots, A_{m/2 + 1,j}, A_{m/2 + 1,n} ), ( A_{m/2 + 2,1}, A_{m/2 + 2,2}, \\cdots, \\\\ A_{m/2 + 2,j}, A_{m/2 + 2,n} ), \\cdots, ( A_{m,1}, A_{m,2}, \\cdots, A_{m,j}, A_{m,n} ) )`

    Examples
    --------
    Import `half_dimension` function from `regressions` library
        >>> from regressions.statistics.halve import half_dimension
    Determine the upper half of the set [[3, 7, 1], [1, 8, 11], [6, 6, 6], [2, 15, 3], [10, 5, 9]] based on the second dimension
        >>> sections_2d = half_dimension([[3, 7, 1], [1, 8, 11], [6, 6, 6], [2, 15, 3], [10, 5, 9]], 2)
        >>> print(sections_2d['upper'])
        [[1, 8, 11], [2, 15, 3]]
    Determine the lower half of the set [[3, 7, 1], [1, 8, 11], [6, 6, 6], [2, 15, 3], [10, 5, 9]] based on the third dimension
        >>> sections_3d = half_dimension([[3, 7, 1], [1, 8, 11], [6, 6, 6], [2, 15, 3], [10, 5, 9]], 3)
        >>> print(sections_3d['lower'])
        [[3, 7, 1], [2, 15, 3]]
    """
    # Handle input errors
    matrix_of_scalars(data, 'first')
    positive_integer(dimension)

    # Sort input according to a provided dimension
    sorted_data = sorted_dimension(data, dimension)

    # Partition sorted input
    result = partition(sorted_data)
    return result