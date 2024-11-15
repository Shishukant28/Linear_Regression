from regressions.errors.vectors import vector_of_scalars
from regressions.errors.scalars import positive_integer
from .minimum import minimum_value
from .maximum import maximum_value
from .median import median_value
from .quartiles import quartile_value
from .rounding import rounded_value

def five_number_summary(data, precision = 4):
    """
    Calculates the five number summary of a given data set: minimum, first quartile, median, third quartile, and maximum

    Parameters
    ----------
    data : list of int or float
        List of numbers to analyze
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the result

    Raises
    ------
    TypeError
        First argument must be a 1-dimensional list
    TypeError
        Elements of first argument must be integers or floats
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    summary['minimum'] : float
        Smallest value from the data set
    summary['q1'] : float
        First quartile of the data set, below which 25% of the data fall
    summary['median'] : float
        Middle value of the data set, splitting the data evenly in half
    summary['q3'] : float
        Third quartile of the data set, above which 25% of the data fall
    summary['maximum'] : float
        Largest value from the data set

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`, :func:`~regressions.statistics.minimum.minimum_value`, :func:`~regressions.statistics.maximum.maximum_value`, :func:`~regressions.statistics.median.median_value`, :func:`~regressions.statistics.quartiles.quartile_value`

    Notes
    -----
    - Set of numbers: :math:`a_i = \\{ a_1, a_2, \\cdots, a_n \\}`
    - Minimum: :math:`a_{min} \\leq a_j, \\forall a_j \\in a_i`
    - Maximum: :math:`a_{max} \\geq a_j, \\forall a_j \\in a_i`
    - For sets with an odd amount of numbers:
        
        - First quartile: :math:`Q_1 = a_{\\lceil n/4 \\rceil}`
        - Median: :math:`M = a_{\\lceil n/2 \\rceil}`
        - Third quartile: :math:`Q_3 = a_{\\lceil 3n/4 \\rceil}`
    
    - For sets with an even amount of numbers:

        - If :math:`n \\text{ mod } 4 \\neq 0`:

            - First quartile: :math:`Q_1 = a_{\\lceil n/4 \\rceil}`
            - Median: :math:`M = \\frac{a_{n/2} + a_{n/2 + 1}}{2}`
            - Third quartile: :math:`Q_3 = a_{\\lceil 3n/4 \\rceil}`
        
        - If :math:`n \\text{ mod } 4 = 0`:

            - First quartile: :math:`Q_1 = \\frac{a_{n/4} + a_{n/4 + 1}}{2}`
            - Median: :math:`M = \\frac{a_{n/2} + a_{n/2 + 1}}{2}`
            - Third quartile: :math:`Q_3 = \\frac{a_{3n/4} + a_{3n/4 + 1}}{2}`

    - |five_number_summary|

    Examples
    --------
    Import `five_number_summary` function from `regressions` library
        >>> from regressions.statistics.summary import five_number_summary
    Determine the five number summary of the set [21, 53, 3, 68, 43, 9, 72, 19, 20, 1]
        >>> summary_even = five_number_summary([21, 53, 3, 68, 43, 9, 72, 19, 20, 1])
        >>> print(summary_even['q1'])
        9.0
        >>> print(summary_even['maximum'])
        72.0
    Determine the five number summary of the set [12, 81, 13, 8, 42, 72, 91, 20, 20]
        >>> summary_odd = five_number_summary([12, 81, 13, 8, 42, 72, 91, 20, 20])
        >>> print(summary_odd['q3'])
        76.5
        >>> print(summary_odd['minimum'])
        8.0
    """
    # Handle input errors
    vector_of_scalars(data, 'first')
    positive_integer(precision)

    # Calculate all values used in five-number summary
    min_value = minimum_value(data)
    q1 = quartile_value(data, 1)
    median = median_value(data)
    q3 = quartile_value(data, 3)
    max_value = maximum_value(data)

    # Round all values
    rounded_min = rounded_value(min_value, precision)
    rounded_q1 = rounded_value(q1, precision)
    rounded_med = rounded_value(median, precision)
    rounded_q3 = rounded_value(q3, precision)
    rounded_max = rounded_value(max_value, precision)
    
    # Package values in single dictionary to return
    result = {
        'minimum': rounded_min,
        'q1': rounded_q1,
        'median': rounded_med,
        'q3': rounded_q3,
        'maximum': rounded_max
    }
    return result