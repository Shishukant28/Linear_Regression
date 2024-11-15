from regressions.errors.matrices import matrix_of_scalars

def transposed_matrix(matrix):
    """
    Transpose a matrix's rows and columns

    Parameters
    ----------
    matrix : list of lists of int or float
        List of lists of numbers representing a matrix

    Raises
    ------
    TypeError
        Argument must be a 2-dimensional list
    TypeError
        Elements nested within argument must be integers or floats
    
    Returns
    -------
    matrix : list of lists of int or float
        List of lists in which each inner element occupies the row that correspond's to the column it occupied in the original matrix and the column that correspond's to the row it occupied in the original matrix

    See Also
    --------
    :func:`~regressions.matrices.cofactors.matrix_of_cofactors`, :func:`~regressions.matrices.minors.matrix_of_minors`, :func:`~regressions.matrices.determinant.linear_determinant`, :func:`~regressions.matrices.inverse.inverse_matrix`

    Notes
    -----
    - Original matrix: :math:`\\mathbf{A} = \\begin{bmatrix} a_{1,1} & a_{1,2} & \\cdots & a_{1,n} \\\\ a_{2,1} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Transpose of matrix: :math:`\\mathbf{A}^T = \\begin{bmatrix} a_{1,1} & a_{2,1} & \\cdots & a_{m,1} \\\\ a_{1,2} & a_{2,2} & \\cdots & a_{m,2} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{1,n} & a_{2,n} & \\cdots & a_{m,n} \\end{bmatrix}`
    - |adjugate|

    Examples
    --------
    Import `transposed_matrix` function from `regressions` library
        >>> from regressions.matrices.transpose import transposed_matrix
    Transpose [[1, 2, 3], [4, 5, 6]]
        >>> matrix_3x2 = transposed_matrix([[1, 2, 3], [4, 5, 6]])
        >>> print(matrix_3x2)
        [[1, 4], [2, 5], [3, 6]]
    Transpose [[2, 3], [5, 7]]
        >>> matrix_2x2 = transposed_matrix([[2, 3], [5, 7]])
        >>> print(matrix_2x2)
        [[2, 5], [3, 7]]
    """
    # Handle input errors
    matrix_of_scalars(matrix)

    # Create list to return
    result = []

    # Iterate over inner lists of input
    for m in range(len(matrix[0])):
        # Create new lists inside list to return
        result.append([])

        # Iterate over outer lists of input
        for n in range(len(matrix)):
            # Store elements previously at inner-outer, now at outer-inner
            result[m].append(matrix[n][m])
    
    # Return result
    return result