from regressions.errors.vectors import vector_of_scalars
from regressions.statistics.summation import sum_value

def vector_magnitude(vector):
    """
    Calculates the magnitude of a vector

    Parameters
    ----------
    vector : list of int or float
        List of numbers representing a vector

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional list
    TypeError
        Elements of argument must be integers or floats

    Returns
    -------
    magnitude : float
        Measure of the size of the vector, as determined by taking the root of the sum of the squares of its components

    See Also
    --------
    :func:`~regressions.vectors.components.component_form`, :func:`~regressions.vectors.direction.vector_direction`,
    :func:`~regressions.vectors.unit.unit_vector`

    Notes
    -----
    - Vector: :math:`\\mathbf{a} = \\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Magnitude of vector: :math:`\\|\\mathbf{a}\\| = \\sqrt{a_1^2 + a_2^2 + \\cdots + a_n^2}`
    - |magnitude|

    Examples
    --------
    Import `vector_magnitude` function from `regressions` library
        >>> from regressions.vectors.magnitude import vector_magnitude
    Determine the magnitude of the vector with components [7, 5, -1]
        >>> magnitude_3d = vector_magnitude([7, 5, -1])
        >>> print(magnitude_3d)
        8.660254037844387
    Determine the magnitude of the vector with components [3, 2]
        >>> magnitude_2d = vector_magnitude([3, 2])
        >>> print(magnitude_2d)
        3.605551275463989
    """
    # Handle input errors
    vector_of_scalars(vector)

    # Create intermediary list
    squares = []

    # Iterate over input
    for element in vector:
        # Store squares of each element in intermediary list
        squares.append(element**2)
    
    # Add all squares in list
    sum_squares = sum_value(squares)

    # Take the square root of the sum of all squares
    result = sum_squares**(1/2)
    return result