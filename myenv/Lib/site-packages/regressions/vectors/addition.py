from regressions.errors.vectors import compare_vectors

def vector_sum(vector_one, vector_two):
    """
    Calculates the sum of two vectors

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
    vector : list of int or float
        List in which each element is the sum of the corresponding elements from the input vectors

    See Also
    --------
    :func:`~regressions.matrices.addition.matrix_sum`, :func:`~regressions.vectors.multiplication.scalar_product_vector`

    Notes
    -----
    - First vector: :math:`\\mathbf{a} = \\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Second vector: :math:`\\mathbf{b} = \\langle b_1, b_2, \\cdots, b_n \\rangle`
    - Sum of vectors: :math:`\\mathbf{a} + \\mathbf{b} = \\langle a_1 + b_1, a_2 + b_2, \\cdots, a_n + b_n \\rangle`
    - |vector_addition|

    Examples
    --------
    Import `vector_sum` function from `regressions` library
        >>> from regressions.vectors.addition import vector_sum
    Add [1, 2, 3] and [4, 5, 6]
        >>> vector_3d = vector_sum([1, 2, 3], [4, 5, 6])
        >>> print(vector_3d)
        [5, 7, 9]
    Add [-5, 12] and [3, -7]
        >>> vector_2d = vector_sum([-5, 12], [3, -7])
        >>> print(vector_2d)
        [-2, 5]
    """
    # Handle input errors
    compare_vectors(vector_one, vector_two)

    # Create list to return
    result = []

    # Iterate over first input
    for i in range(len(vector_one)):
        # Store sums of corresponding vector elements in result
        result.append(vector_one[i] + vector_two[i])
    
    # Return result
    return result