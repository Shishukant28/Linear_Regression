from regressions.errors.matrices import matrix_of_scalars, level
from regressions.errors.scalars import positive_integer

def single_dimension(matrix, scalar = 1):
    """
    Extracts a column vector as a row vector from a matrix according to an integer corresponding to the column's position

    Parameters
    ----------
    matrix : list of lists of int or float
        List containing other lists, where each inner list is a row and elements within those inner lists correspond to columns
    scalar : int, default=1
        Number corresponding to the column's position

    Raises
    ------
    TypeError
        First argument must be a 2-dimensional list
    TypeError
        Elements nested within the first argument's lists must be integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    vector : list of int or float
        List containing only integers or floats

    See Also
    --------
    :func:`~regressions.vectors.column.column_conversion`, :func:`~regressions.statistics.sort.sorted_dimension`, :func:`~regressions.statistics.half.half_dimension`

    Notes
    -----
    - Matrix: :math:`\\begin{bmatrix} a_{1,1} & a_{1,2} & \\cdots & a_{1,n} \\\\ a_{2,1} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Row vector corresponding to the :math:`n`\ th column of the matrix: :math:`\\langle a_{1,n}, a_{2,n}, \\cdots, a_{m,n} \\rangle`

    Examples
    --------
    Import `single_dimension` function from `regressions` library
        >>> from regressions.vectors.dimension import single_dimension
    Extract the second column from the matrix [[3, 5, 9], [1, -4, 2]]
        >>> vector_2c = single_dimension([[3, 5, 9], [1, -4, 2]], 2)
        >>> print(vector_2c)
        [5, -4]
    Extract the first column from the matrix [[3, 5, 9], [1, -4, 2]]
        >>> vector_1c = single_dimension([[3, 5, 9], [1, -4, 2]], 1)
        >>> print(vector_1c)
        [3, 1]
    """
    # Handle input errors
    matrix_of_scalars(matrix, 'first')
    positive_integer(scalar)
    level(matrix, scalar)

    # Create list to return
    result = []

    # Iterate over input
    for element in matrix:
        # Store all elements at given dimension in list to return
        result.append(element[scalar - 1])
    
    # Return result
    return result