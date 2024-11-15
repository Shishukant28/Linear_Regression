from math import atan, degrees
from regressions.errors.vectors import vector_of_scalars, length

def vector_direction(vector):
    """
    Calculates the direction of a vector in radians and degrees

    Parameters
    ----------
    vector : list of int or float
        List of two numbers representing a vector, in which the first number is the horizontal component and the second is the vertical component

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional list
    TypeError
        Elements of argument must be integers or floats
    ValueError
        Argument must contain exactly two elements

    Returns
    -------
    direction['radian'] : float
        Measure of the angle of the vector in radians
    direction['degree'] : float
        Measure of the angle of the vector in degrees

    See Also
    --------
    :func:`~regressions.vectors.components.component_form`, :func:`~regressions.vectors.magnitude.vector_magnitude`, :func:`~regressions.vectors.unit.unit_vector`

    Notes
    -----
    - Vector: :math:`\\langle x, y \\rangle`
    - Direction of vector: :math:`\\theta = \\tan^{-1}(\\frac{y}{x})`
    - |direction|

    Examples
    --------
    Import `vector_direction` function from `regressions` library
        >>> from regressions.vectors.direction import vector_direction
    Determine the direction of a vector with a component form of [7, 5]
        >>> direction_positive = vector_direction([7, 5])
        >>> print(direction_positive['radian'])
        0.6202494859828215
        >>> print(direction_positive['degree'])
        35.53767779197438
    Determine the direction of a vector with a component form of [-3, 11]
        >>> direction_negative = vector_direction([-3, 11])
        >>> print(direction_negative['radian'])
        -1.3045442776439713
        >>> print(direction_negative['degree'])
        -74.74488129694222
    """
    # Handle input errors
    vector_of_scalars(vector)
    length(vector, 2)

    # Circumvent division by zero
    if vector[0] == 0:
        vector[0] = 0.0001
    
    # Create intermediary variable
    ratio = vector[1] / vector[0]
    
    # Determine direction in radians
    radian_measure = atan(ratio)

    # Convert radians into degrees
    degree_measure = degrees(radian_measure)
    
    # Package both measures in single dictionary
    result = {
        'radian': radian_measure,
        'degree': degree_measure
    }
    return result