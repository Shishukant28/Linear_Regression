from regressions.errors.scalars import scalar_value
from regressions.errors.vectors import vector_of_scalars, compare_vectors

def scalar_product_vector(vector, scalar):
    """
    Calculates the product of a vector and a scalar

    Parameters
    ----------
    vector : list of int or float
        List of numbers representing a vector
    scalar : int or float
        Number representing a scalar

    Raises
    ------
    TypeError
        First argument must be a 1-dimensional list
    TypeError
        Elements of first argument must be integers or floats
    TypeError
        Second argument must be an integer or a float

    Returns
    -------
    product : list of int or float
        List of numbers in which each element is the product of the scalar factor and the corresponding element from the input vector

    See Also
    --------
    :func:`~regressions.matrices.multiplication.scalar_product_matrix`, :func:`~regressions.vectors.addition.vector_sum`

    Notes
    -----
    - Vector: :math:`\\mathbf{a} = \\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Scalar: :math:`c`
    - Scalar product: :math:`c\\cdot{\\mathbf{a}} = \\langle c\\cdot{a_1}, c\\cdot{a_2}, \\cdots, c\\cdot{a_n} \\rangle`
    - |scalar_multiplication|

    Examples
    --------
    Import `scalar_product_vector` function from `regressions` library
        >>> from regressions.vectors.multiplication import scalar_product_vector
    Multiply [1, 2, 3] and -2
        >>> product_3d = scalar_product_vector([1, 2, 3], -2)
        >>> print(product_3d)
        [-2, -4, -6]
    Multiply [-5, 12] and 3
        >>> product_2d = scalar_product_vector([-5, 12], 3)
        >>> print(product_2d)
        [-15, 36]
    """
    # Handle input errors
    vector_of_scalars(vector, 'first')
    scalar_value(scalar, 'second')

    # Create list to return
    result = []

    # Iterate over input
    for element in vector:
        # Store products in list to return
        result.append(element * scalar)
    
    # Return result
    return result

def dot_product(vector_one, vector_two):
    """
    Calculates the product of two vectors

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
    product : float
        Number created by summing the products of the corresponding terms from each input vector

    See Also
    --------
    :func:`~regressions.matrices.multiplication.matrix_product`

    Notes
    -----
    - First vector: :math:`\\mathbf{a} = \\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Second vector: :math:`\\mathbf{b} = \\langle b_1, b_2, \\cdots, b_n \\rangle`:
    - Dot product of vectors: :math:`\\mathbf{a}\\cdot{\\mathbf{b}} = a_1\\cdot{b_1} + a_2\\cdot{b_2} + \\cdots + a_n\\cdot{b_n}`
    - |dot_product|

    Examples
    --------
    Import `dot_product` function from `regressions` library
        >>> from regressions.vectors.multiplication import dot_product
    Multiply [1, 2, 3] and [4, 5, 6]
        >>> product_3d = dot_product([1, 2, 3], [4, 5, 6])
        >>> print(product_3d)
        32.0
    Multiply [-5, 12] and [3, -7]
        >>> product_2d = dot_product([-5, 12], [3, -7])
        >>> print(product_2d)
        -99.0
    """
    # Handle input errors
    compare_vectors(vector_one, vector_two)

    # Create intermediary number
    result = 0

    # Iterate over inputs
    for i in range(len(vector_one)):
        # Add products of corresponding elements from inputs to intermediary number
        result += vector_one[i] * vector_two[i]
    
    # Convert number to float
    floated_result = float(result)
    return floated_result