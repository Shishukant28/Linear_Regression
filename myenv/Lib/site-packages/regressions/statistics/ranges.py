from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import four_scalars, compare_scalars
from .maximum import maximum_value
from .minimum import minimum_value

def range_value(data):
    """
    Determines the range of a data set (i.e., the difference between its largest value and its smallest value)

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze

    Raises
    ------
    TypeError
        Argument must be a 1-dimensional list
    TypeError
        Elements of argument must be integers or floats

    Returns
    -------
    interval : float
        Range of data set

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`, :func:`~regressions.statistics.minimum.minimum_value`, :func:`~regressions.statistics.maximum.maximum_value`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Range of set: :math:`R = a_{max} - a_{min}`
    
        - :math:`a_{min} \\leq a_j, \\forall a_j \\in a_i`
        - :math:`a_{max} \\geq a_j, \\forall a_j \\in a_i`

    - |range|

    Examples
    --------
    Import `range_value` function from `regressions` library
        >>> from regressions.statistics.ranges import range_value
    Determine the range of the set [21, 53, 3, 68, 43, 9, 72, 19, 20, 1]
        >>> range_even = range_value([21, 53, 3, 68, 43, 9, 72, 19, 20, 1])
        >>> print(range_even)
        71.0
    Determine the range of the set [12, 81, 13, 8, 42, 72, 91, 20, 20]
        >>> range_odd = range_value([12, 81, 13, 8, 42, 72, 91, 20, 20])
        >>> print(range_odd)
        83.0
    """
    # Handle input errors
    vector_of_scalars(data)

    # Determine maximum and minimum of input
    max_value = maximum_value(data)
    min_value = minimum_value(data)

    # Calculate difference between maximum and minimum
    difference = max_value - min_value

    # Convert difference to float
    result = float(difference)
    return result

def shift_into_range(initial_value, periodic_unit, minimum, maximum):
    """
    Adjusts an intial value to one within a particular range by increasing or decreasing its value by a specified unit

    Parameters
    ----------
    initial_value : int or float
        Starting value to adjust to fit into a range
    periodic_unit : int or float
        Unit by which the initial value should be incrementally increased or decreased to fit into a range
    minimum : int or float
        Lower bound of range into which the initial value must be adjusted (final value should be greater than or equal to minimum)
    maximum : int or float
        Upper bound of range into which the initial value must be adjusted (final value should be less than or equal to maximum)

    Raises
    ------
    TypeError
        Arguments must be integers or floats
    ValueError
        Third argument must be less than or equal to fourth argument

    Returns
    -------
    final_value : float
        Value within range that only differs from the initial value by a an integral multiple of the periodic unit

    See Also
    --------
    :func:`~regressions.analyses.points.shifted_points_within_range`, :func:`~regressions.analyses.mean_values.mean_values_derivative`, :func:`~regressions.analyses.mean_values.mean_values_integral`

    Notes
    -----
    - Initial value: :math:`v_i`
    - Periodic unit: :math:`\\lambda`
    - Lower bound of range: :math:`b_l`
    - Upper bound of range: :math:`b_u`
    - Set of all values derived from initial value and periodic unit: :math:`g = \\{ v \\mid v = v_i + \\lambda\\cdot{k} \\}`

        - :math:`k \\in \\mathbb{Z}`
    
    - Final value: :math:`v_f \\geq b_l \\cap v_f \\leq b_u \\cap v_f \\in g`

    Examples
    --------
    Import `shift_into_range` function from `regressions` library
        >>> from regressions.statistics.ranges import shift_into_range
    Adjust the number 7 to a value between 20 and 30, based on a periodic unit of 8
        >>> final_value_int = shift_into_range(7, 8, 20, 30)
        >>> print(final_value_int)
        23.0
    Adjust the number 524.62 to a value between 138.29 and 213.86, based on a periodic unit of 23.91
        >>> final_value_float = shift_into_range(524.62, 23.91, 138.29, 213.86)
        >>> print(final_value_float)
        213.78999999999974
    """
    # Handle input errors
    four_scalars(initial_value, periodic_unit, minimum, maximum)
    compare_scalars(minimum, maximum, 'third', 'fourth')

    # Set input value to alternative value
    alternative_initial_value = initial_value
    
    # Handle positive periodic units
    if periodic_unit > 0:
        # Decrease value till below maximum
        while alternative_initial_value > maximum:
            alternative_initial_value -= periodic_unit

        # Increase value till above minimum
        while alternative_initial_value < minimum:
            alternative_initial_value += periodic_unit
    
    # Handle negative periodic units
    else:
        # Decrease value till below maximum
        while alternative_initial_value > maximum:
            alternative_initial_value += periodic_unit

        # Increase value till above minimum
        while alternative_initial_value < minimum:
            alternative_initial_value -= periodic_unit

    # Convert final value to float
    final_value = float(alternative_initial_value)
    return final_value