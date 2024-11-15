from regressions.errors.matrices import compare_rows
from regressions.errors.scalars import positive_integer
from regressions.statistics.rounding import rounded_list
from regressions.vectors.dimension import single_dimension
from .multiplication import matrix_product
from .transpose import transposed_matrix
from .inverse import inverse_matrix

def system_solution(matrix_one, matrix_two, precision = 4):
    """
    Solves a system of equations using matrices (independent matrix multiplied by resultant matrix equals dependent matrix)

    Parameters
    ----------
    matrix_one : list of lists of int or float
        List of lists of numbers representing the independent matrix of a system of equations
    matrix_two : list of lists of int or float
        List of lists of numbers representing the dependent matrix of a system of equations
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the results

    Raises
    ------
    TypeError
        First and second arguments must be 2-dimensional lists
    TypeError
        Elements nested within first and second arguments must be integers or floats
    ValueError
        First and second arguments must contain the same amount of lists
    ValueError
        Last argument must be a positive integer
    
    Returns
    -------
    solution : list of float
        Row vector of coefficients that if expressed as a column vector would satisfy the equation

    See Also
    --------
    :func:`~regressions.matrices.multiplication.matrix_product`, :func:`~regressions.matrices.transpose.transposed_matrix`, :func:`~regressions.matrices.determinant.linear_determinant`, :func:`~regressions.matrices.inverse.inverse_matrix`

    Notes
    -----
    - Independent matrix: :math:`\\mathbf{A} = \\begin{bmatrix} a_{1,1} & a_{1,2} & \\cdots & a_{1,n} \\\\ a_{2,1} & a_{2,2} & \\cdots & a_{2,n} \\\\ \\cdots & \\cdots & \\cdots & \\cdots \\\\ a_{m,1} & a_{m,2} & \\cdots & a_{m,n} \\end{bmatrix}`
    - Dependent matrix: :math:`\\mathbf{B} = \\begin{bmatrix} b_{1,1} \\\\ b_{2,1} \\\\ \\cdots \\\\ b_{m,1} \\end{bmatrix}`
    - Variable matrix: :math:`\\mathbf{X} = \\begin{bmatrix} x_{1,1} \\\\ x_{2,1} \\\\ \\cdots \\\\ x_{m,1} \\end{bmatrix}`
    - System of equations in terms of matrices: :math:`\\mathbf{A}\\cdot{\\mathbf{X}} = \\mathbf{B}`
    - Solution of system of equations: :math:`\\mathbf{X} = \\mathbf{A}^{-1}\\cdot{\\mathbf{B}}`
    - |solve|

    Examples
    --------
    Import `system_solution` function from `regressions` library
        >>> from regressions.matrices.solve import system_solution
    Solve the system that has an independent matrix of [[2, 3], [1, -1]] and a dependent matrix of [[5], [1]]
        >>> solution_2values = system_solution([[2, 3], [1, -1]], [[5], [1]])
        >>> print(solution_2values)
        [1.6, 0.6]
    Solve the system that has an independent matrix of [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, 12]] and a dependent matrix of [[2], [-3], [5], [-7]]
        >>> solution_3values = system_solution([[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, 12]], [[2], [-3], [5], [-7]])
        >>> print(solution_3values)
        [-0.8611, -1.3889, -0.0278]
    """
    # Handle input errors
    compare_rows(matrix_one, matrix_two)
    positive_integer(precision)

    # Multiply inverse of first matrix by second matrix
    transposition = transposed_matrix(matrix_one)
    product = matrix_product(transposition, matrix_one)
    inversion = inverse_matrix(product)
    second_product = matrix_product(inversion, transposition)
    solution_column = matrix_product(second_product, matrix_two)

    # Convert solution from a column vector into a row vector
    solution = single_dimension(solution_column, 1)

    # Round result
    result = rounded_list(solution, precision)
    return result