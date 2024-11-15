from regressions.errors.matrices import compare_matrices

def matrix_sum(matrix_one, matrix_two):
    """
    Calculates the sum of two matrices

    Parameters
    ----------
    matrix_one : list of lists of int or float
        List of lists of numbers representing a matrix
    matrix_two : list of lists of int or float
        List of lists of numbers representing a matrix

    Raises
    ------
    TypeError
        Arguments must be 2-dimensional lists
    TypeError
        Elements nested within arguments must be integers or floats
    ValueError
        Both arguments must contain the same amount of lists
    ValueError
        First list within first argument and first list within second argument must contain the same amount of elements

    Returns
    -------
    matrix : list of lists of int or float
        List of lists in which each inner element is the sum of the corresponding elements from the input matrices
    
    See Also
    --------
    :func:`~regressions.vectors.addition.vector_sum`, :func:`~regressions.matrices.multiplication.scalar_product_matrix`

    Notes
    -----
    - First matrix: :math:`\\mathbf{A} = \\begin{bmatrix} a_{1,1} & a_{1,2} & \\cdots & a_{1,n} \\\\ a_{2,1} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Second matrix: :math:`\\mathbf{B} = \\begin{bmatrix} b_{1,1} & b_{1,2} & \\cdots & b_{1,n} \\\\ b_{2,1} & b_{2,2} & \\cdots & b_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ b_{m,1} & b_{m,2} & \\cdots & b_{m,n} \\end{bmatrix}`
    - Sum of matrices: :math:`\\mathbf{A} + \\mathbf{B} = \\begin{bmatrix} a_{1,1} + b_{1,1} & a_{1,2} + b_{1,2} & \\cdots & a_{1,n} + b_{1,n} \\\\ a_{2,1} + b_{2,1} & a_{2,2} + b_{2,2} & \\cdots & a_{2,n} + b_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} + b_{m,1} & a_{m,2} + b_{m,2} & \\cdots & a_{m,n} + b_{m,n} \\end{bmatrix}`
    - |matrix_addition|

    Examples
    --------
    Import `matrix_sum` function from `regressions` library
        >>> from regressions.matrices.addition import matrix_sum
    Add [[1, 2, 3], [4, 5, 6]] and [[2, 3, 5], [7, 11, 13]]
        >>> matrix_2x3 = matrix_sum([[1, 2, 3], [4, 5, 6]], [[2, 3, 5], [7, 11, 13]])
        >>> print(matrix_2x3)
        [[3, 5, 8], [11, 16, 19]]
    Add [[-2, 5], [7, -1]] and [[8, 2], [-3, 4]]
        >>> matrix_2x2 = matrix_sum([[-2, 5], [7, -1]], [[8, 2], [-3, 4]])
        >>> print(matrix_2x2)
        [[6, 7], [4, 3]]
    """
    # Handle input errors
    compare_matrices(matrix_one, matrix_two)

    # Create list to return
    result = []

    # Iterate over outer lists of input
    for m in range(len(matrix_one)):
        # Create new lists inside list to return
        result.append([])

        # Iterate over inner lists of input
        for n in range(len(matrix_one[0])):
            # Store sums in inner lists of return
            result[m].append(matrix_one[m][n] + matrix_two[m][n])
    
    # Return result
    return result