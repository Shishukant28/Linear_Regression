from regressions.errors.matrices import square_matrix
from .multiplication import scalar_product_matrix
from .determinant import linear_determinant
from .transpose import transposed_matrix
from .minors import matrix_of_minors
from .cofactors import matrix_of_cofactors

def inverse_matrix(matrix):
    """
    Generate the inverse matrix of a given matrix

    Parameters
    ----------
    matrix : list of lists of int or float
        List of lists of numbers representing a matrix

    Raises
    ------
    TypeError
        First argument must be a 2-dimensional list
    TypeError
        Elements nested within first argument must be integers or floats
    ValueError
        First argument must contain the same amount of lists as the amount of elements contained within its first list
    
    Returns
    -------
    inverse : list of lists of float
        List of lists corresponding to the inverse of the original matrix; if original matrix has a determinant of zero, then 0.0001 will be used as its determinant, ensuring a result

    See Also
    --------
    :func:`~regressions.matrices.cofactors.matrix_of_cofactors`, :func:`~regressions.matrices.minors.matrix_of_minors`, :func:`~regressions.matrices.transpose.transposed_matrix`, :func:`~regressions.matrices.determinant.linear_determinant`, :func:`~regressions.matrices.multiplication.scalar_product_matrix`, :func:`~regressions.matrices.solve.system_solution`

    Notes
    -----
    - Original matrix: :math:`\\mathbf{A} = \\begin{bmatrix} a_{1,1} & a_{1,2} & \\cdots & a_{1,n} \\\\ a_{2,1} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Inverse of matrix: :math:`\\mathbf{A}^{-1} = \\frac{1}{|\\mathbf{A}|}\\cdot{{{\\mathbf{A}^M}^C}^T}`
    - |inverse|

    Examples
    --------
    Import `inverse_matrix` function from `regressions` library
        >>> from regressions.matrices.inverse import inverse_matrix
    Generate the inverse of [[1, 2], [3, 4]]
        >>> inverse_2x2 = inverse_matrix([[1, 2], [3, 4]])
        >>> print(inverse_2x2)
        [[-2.0, 1.0], [1.5, -0.5]]
    Generate the inverse of [[2, 3, 5], [7, 11, 13], [17, 19, 23]]
        >>> inverse_3x3 = inverse_matrix([[2, 3, 5], [7, 11, 13], [17, 19, 23]])
        >>> print(inverse_3x3)
        [[-0.07692307692307693, -0.3333333333333333, 0.20512820512820512], [-0.7692307692307692, 0.5, -0.11538461538461538], [0.6923076923076923, -0.16666666666666666, -0.01282051282051282]]
    """
    # Handle input errors
    square_matrix(matrix)

    # Determine determinant of matrix
    determinant = linear_determinant(matrix)

    # Circumvent division by zero
    if determinant == 0:
        determinant = 0.0001
    
    # Calculate reciprocal of determinant
    determinant_reciprocal = 1 / determinant
    
    # Create adjugate of original matrix by transposing its cofactors
    adjugate = transposed_matrix(matrix_of_cofactors(matrix_of_minors(matrix)))

    # Generate inverse matrix by multiplying adjugate by reciprocal
    result = scalar_product_matrix(adjugate, determinant_reciprocal)
    return result