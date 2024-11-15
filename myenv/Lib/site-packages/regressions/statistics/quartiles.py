from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import select_integers
from .median import median_value
from .halve import half

def quartile_value(data, q):
    """
    Determines the first, second, or third quartile values of a data set

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze
    q : int
        Number determining which quartile to provide

    Raises
    ------
    TypeError
        First argument must be a 1-dimensional list
    TypeError
        Elements of first argument must be integers or floats
    ValueError
        Second argument must be an integer contained within the set [1, 2, 3]

    Returns
    -------
    quartile : int or float
        Quartile value of the data set

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`, :func:`~regressions.statistics.halve.half`, :func:`~regressions.statistics.minimum.minimum_value`, :func:`~regressions.statistics.maximum.maximum_value`, :func:`~regressions.statistics.median.median_value`

    Notes
    -----
    - Ordered set of numbers: :math:`a_i = ( a_1, a_2, \\cdots, a_n )`
    - For sets with an odd amount of numbers:
        
        - First quartile: :math:`Q_1 = a_{\\lceil n/4 \\rceil}`
        - Second quartile: :math:`Q_2 = a_{\\lceil n/2 \\rceil}`
        - Third quartile: :math:`Q_3 = a_{\\lceil 3n/4 \\rceil}`
    
    - For sets with an even amount of numbers:

        - If :math:`n \\text{ mod } 4 \\neq 0`:

            - First quartile: :math:`Q_1 = a_{\\lceil n/4 \\rceil}`
            - Second quartile: :math:`Q_2 = \\frac{a_{n/2} + a_{n/2 + 1}}{2}`
            - Third quartile: :math:`Q_3 = a_{\\lceil 3n/4 \\rceil}`
        
        - If :math:`n \\text{ mod } 4 = 0`:

            - First quartile: :math:`Q_1 = \\frac{a_{n/4} + a_{n/4 + 1}}{2}`
            - Second quartile: :math:`Q_2 = \\frac{a_{n/2} + a_{n/2 + 1}}{2}`
            - Third quartile: :math:`Q_3 = \\frac{a_{3n/4} + a_{3n/4 + 1}}{2}`

    - |quartiles|

    Examples
    --------
    Import `quartile_value` function from `regressions` library
        >>> from regressions.statistics.quartiles import quartile_value
    Determine the first quartile of the set [21, 53, 3, 68, 43, 9, 72, 19, 20, 1]
        >>> quartile_1 = quartile_value([21, 53, 3, 68, 43, 9, 72, 19, 20, 1], 1)
        >>> print(quartile_1)
        9
    Determine the third quartile of the set [12, 81, 13, 8, 42, 72, 91, 20, 20]
        >>> quartile_3 = quartile_value([12, 81, 13, 8, 42, 72, 91, 20, 20], 3)
        >>> print(quartile_3)
        76.5
    """
    # Handle input errors
    vector_of_scalars(data, 'first')
    select_integers(q, [1, 2, 3])

    # Split input in half
    halved_data = half(data)

    # Create number to return
    result = 0

    # Determine Q2 by taking the median of all elements
    if q == 2:
        result = median_value(data)
    
    # Determine Q1 by taking the median of the lower half of elements
    elif q == 1:
        result = median_value(halved_data['lower'])
    
    # Determine Q3 by taking the median of the upper half of elements
    elif q == 3:
        result = median_value(halved_data['upper'])
    
    # Return result
    return result