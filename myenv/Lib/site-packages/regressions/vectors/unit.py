from regressions.errors.vectors import vector_of_scalars
from .magnitude import vector_magnitude
from .multiplication import scalar_product_vector

def unit_vector(vector):
    """
    Calculates the unit vector corresponding to a given vector (and therefore having the same direction)

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
    unit : list of float
        Vector with a magnitue of 1 in the same direction as the original vector

    See Also
    --------
    :func:`~regressions.vectors.components.component_form`, :func:`~regressions.vectors.multiplication.scalar_product_vector`, :func:`~regressions.vectors.direction.vector_direction`,
    :func:`~regressions.vectors.magnitude.vector_magnitude`

    Notes
    -----
    - Comparison vector: :math:`\\mathbf{a} = \\langle a_1, a_2, \\cdots, a_n \\rangle`
    - Unit vector with same direction: :math:`\\mathbf{u}= \\frac{\\mathbf{a}}{\\|\\mathbf{a}\\|}`
    - |unit_vector|

    Examples
    --------
    Import `unit_vector` function from `regressions` library
        >>> from regressions.vectors.unit import unit_vector
    Determine the unit vector of the vector with components [7, 5, -1]
        >>> unit_3d = unit_vector([7, 5, -1])
        >>> print(unit_3d)
        [0.8082903768654759, 0.5773502691896257, -0.11547005383792514]
    Determine the unit vector of the vector with components [3, 2]
        >>> unit_2d = unit_vector([3, 2])
        >>> print(unit_2d)
        [0.8320502943378437, 0.5547001962252291]
    """
    # Handle input errors
    vector_of_scalars(vector)

    # Determine magnitude of input
    magnitude = vector_magnitude(vector)

    # Circumvent division by zero
    if magnitude == 0:
        magnitude = 0.0001
    
    # Calculate reciprocal of magnitude
    reciprocal_magnitude = 1 / magnitude

    # Multiply input by the reciprocal of its magnitude
    result = scalar_product_vector(vector, reciprocal_magnitude)
    return result