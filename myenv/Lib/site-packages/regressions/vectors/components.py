from regressions.errors.vectors import compare_vectors
from .addition import vector_sum
from .multiplication import scalar_product_vector

def component_form(initial_point, terminal_point):
    """
    Calculates the component form for a vector by using two points

    Parameters
    ----------
    initial_point : list of int or float
        List of numbers representing a point
    terminal_point : list of int or float
        List of numbers representing a point

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
    components : list of int or float
        List in which each element is the difference of the corresponding elements from the input points (specifically, the change from the initial point to the terminal point)

    See Also
    --------
    :func:`~regressions.vectors.addition.vector_sum`, :func:`~regressions.vectors.multiplication.scalar_product_vector`, :func:`~regressions.vectors.direction.vector_direction`, :func:`~regressions.vectors.magnitude.vector_magnitude`

    Notes
    -----
    - Initial point: :math:`A = (a_1, a_2, \\cdots, a_n)`
    - Terminal point: :math:`B = (b_1, b_2, \\cdots, b_n)`
    - Component form of vector: :math:`\\overrightarrow{AB} = \\langle b_1 - a_1, b_2 - a_2, \\cdots, b_n - a_n \\rangle`
    - |component_form|

    Examples
    --------
    Import `component_form` function from `regressions` library
        >>> from regressions.vectors.components import component_form
    Determine the component form of a vector with an initial point of [1, 2, 3] and a terminal point of [4, 5, 6]
        >>> components_3d = component_form([1, 2, 3], [4, 5, 6])
        >>> print(components_3d)
        [3, 3, 3]
    Determine the component form of a vector with an initial point of [-5, 12] and a terminal point of [3, -7]
        >>> components_2d = component_form([-5, 12], [3, -7])
        >>> print(components_2d)
        [8, -19]
    """
    # Handle input errors
    compare_vectors(initial_point, terminal_point)

    # Determine difference between terminal point and initial point
    result = vector_sum(terminal_point, scalar_product_vector(initial_point, -1))
    return result