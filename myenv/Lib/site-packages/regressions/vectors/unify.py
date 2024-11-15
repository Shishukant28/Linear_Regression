from regressions.errors.vectors import compare_vectors

def unite_vectors(vector_one, vector_two):
    """
    Unites two row vectors into a single matrix whose columns coincide with the input vectors

    Parameters
    ----------
    vector_one : list of int or float
        List of numbers representing a vector
    vector_two : list of int or float
        List of numbers representing a vector

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
    matrix : list of lists of int or float
        List containing lists; length of outer list will equal lengths of supplied vectors; length of inner lists will equal two

    See Also
    --------
    :func:`~regressions.vectors.dimension.single_dimension`, :func:`~regressions.statistics.halve.partition`

    Notes
    -----
    - First vector: :math:`\\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Second vector: :math:`\\langle b_1, b_2, \\cdots, b_n \\rangle`
    - Matrix unifying first and second vectors: :math:`\\begin{bmatrix} a_1 & b_1 \\\\ a_2 & b_2 \\\\ \\cdots & \\cdots \\\\ a_n & b_n \\end{bmatrix}`

    Examples
    --------
    Import `unite_vectors` function from `regressions` library
        >>> from regressions.vectors.unify import unite_vectors
    Unite [1, 2, 3] and [4, 5, 6]
        >>> matrix_3x2 = unite_vectors([1, 2, 3], [4, 5, 6])
        >>> print(matrix_2x3)
        [[1, 4], [2, 5], [3, 6]]
    Unite [-5, 12] and [3, -7]
        >>> matrix_2x2 = unite_vectors([-5, 12], [3, -7])
        >>> print(matrix_2x2)
        [[-5, 3], [12, -7]]
    """
    # Handle input errors
    compare_vectors(vector_one, vector_two)

    # Create list to return
    result = []

    # Handle no solution
    if vector_one[0] == None:
        result.append(None)
    
    # Handle general case
    else:
        # Iterate over inputs
        for i in range(len(vector_one)):
            # Store corresponding elements from inputs as lists within list to return
            result.append([vector_one[i], vector_two[i]])
    
    # Return result
    return result