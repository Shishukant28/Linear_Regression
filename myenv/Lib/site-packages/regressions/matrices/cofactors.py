from regressions.errors.matrices import matrix_of_scalars

def matrix_of_cofactors(matrix):
    """
    Create the matrix of cofactors corresponding to a given matrix

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
        List of lists in which each inner element alternates being positive or negative versions of the corresponding element from the original matrix

    See Also
    --------
    :func:`~regressions.matrices.minors.matrix_of_minors`, :func:`~regressions.matrices.transpose.transposed_matrix`, :func:`~regressions.matrices.determinant.linear_determinant`, :func:`~regressions.matrices.inverse.inverse_matrix`

    Notes
    -----
    - Original matrix: :math:`\\mathbf{A} = \\begin{bmatrix} a_{1,1} & a_{1,2} & \\cdots & a_{1,n} \\\\ a_{2,1} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Matrix of cofactors (if :math:`\\mathbf{A}` contains an odd number of rows and columns): :math:`\\mathbf{A}^C = \\begin{bmatrix} a_{1,1} & -1\\cdot{a_{1,2}} & \\cdots & a_{1,n} \\\\ -1\\cdot{a_{2,1}} & a_{2,2} & \\cdots & -1\\cdot{a_{2,n}} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & -1\\cdot{a_{m,2}} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Matrix of cofactors (if :math:`\\mathbf{A}` contains an even number of rows and columns): :math:`\\mathbf{A}^C = \\begin{bmatrix} a_{1,1} & -1\\cdot{a_{1,2}} & \\cdots & -1\\cdot{a_{1,n}} \\\\ -1\\cdot{a_{2,1}} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ -1\\cdot{a_{m,1}} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - |cofactors|

    Examples
    --------
    Import `matrix_of_cofactors` function from `regressions` library
        >>> from regressions.matrices.cofactors import matrix_of_cofactors
    Create the matrix of cofactors for [[1, 2, 3], [4, 5, 6]]
        >>> matrix_3x2 = matrix_of_cofactors([[1, 2, 3], [4, 5, 6]])
        >>> print(matrix_3x2)
        [[1, -2, 3], [-4, 5, -6]]
    Create the matrix of cofactors for [[2, 3], [5, 7]]
        >>> matrix_2x2 = matrix_of_cofactors([[2, 3], [5, 7]])
        >>> print(matrix_2x2)
        [[2, -3], [-5, 7]]
    """
    # Handle input errors
    matrix_of_scalars(matrix)

    # Create list to return
    result = []

    # Iterate over outer lists of input
    for m in range(len(matrix)):
        # Create new lists inside list to return
        result.append([])

        if m % 2 == 0:
            # Iterate over inner lists of input
            for n in range(len(matrix[0])):
                # Handle even-even indexed elements
                if n % 2 == 0:
                    result[m].append(matrix[m][n])
                
                # Handle even-odd indexed elements
                else:
                    result[m].append(-1 * matrix[m][n])
        
        else:
            # Iterate over inner lists of input
            for n in range(len(matrix[0])):
                # Handle odd-even indexed elements
                if n % 2 == 0:
                    result[m].append(-1 * matrix[m][n])
                
                # Handle odd-odd indexed elements
                else:
                    result[m].append(matrix[m][n])
    
    # Return result
    return result