from regressions.errors.vectors import vector_of_scalars

def column_conversion(vector):
    """
    Converts a row vector into a column vector

    Parameters
    ----------
    vector : list of int or float
        List of numbers representing a vector

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional list

    Returns
    -------
    column : list of lists of int or float
        List in which each element is a list containing an element from the input vector

    See Also
    --------
    :func:`~regressions.vectors.dimension.single_dimension`

    Notes
    -----
    - Row vector: :math:`\\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Column vector: :math:`\\left\\langle\\begin{matrix} a_1, \\\\ a_2, \\\\ \\cdots, \\\\ a_n \\end{matrix}\\right\\rangle`

    Examples
    --------
    Import `column_conversion` function from `regressions` library
        >>> from regressions.vectors.column import column_conversion
    Convert [1, 2, 3]
        >>> column_3d = column_conversion([1, 2, 3])
        >>> print(column_3d)
        [[1], [2], [3]]
    Convert [-7, 5]
        >>> column_2d = column_conversion([-7, 5])
        >>> print(column_2d)
        [[-7], [5]]
    """
    # Handle input errors
    vector_of_scalars(vector)

    # Create list to return
    result = []

    # Iterate over input
    for element in vector:
        # Store elements as lists within list to return
        result.append([element])
    
    # Return result
    return result