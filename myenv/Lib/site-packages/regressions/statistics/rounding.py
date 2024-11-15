from regressions.errors.scalars import allow_none_scalar, positive_integer
from regressions.errors.vectors import allow_none_vector

def rounded_value(number, precision = 4):
    """
    Rounds a number to a certain decimal place, but returns a non-zero result if the number being rounded is non-zero even if it would round to zero at that level of decimal precision; allows None as a possible input

    Parameters
    ----------
    number : int or float
        Number to round
    precision : int, default=4
        Maximum number of digits that can appear after the decimal place of the result

    Raises
    ------
    TypeError
        First argument must be an integer, a float, or None
    ValueError
        Last argument must be a positive integer

    Returns
    -------
    number : float
        Original number rounded to the number of decimal places indicated by the precision input

    See Also
    --------
    :func:`~regressions.statistics.sort.sorted_list`, :func:`~regressions.statistics.summation.sum_value`

    Notes
    -----
    - Number to round: :math:`n`
    - Absolute value of number: :math:`a = |n|`
    - Maximum number of digits after decimal place of result: :math:`d`
    - Check size of number: :math:`c(a,d) = \\lfloor a\\cdot{10^d} \\rfloor`
    - Significant digit: :math:`s(a,d) = \\lfloor ( a\\cdot{10^d} - \\lfloor a\\cdot{10^d} \\rfloor )\\cdot{10} \\rfloor`
    - If :math:`c(a,d) = 0`:
        
        - Rounding formula (if :math:`n = 0`): :math:`r = 0`
        - If :math:`n \\neq 0`:

            - Rounding formula (if :math:`a = n`): :math:`r(d) = 10^{-d}`
            - Rounding formula (if :math:`a \\neq n`): :math:`r(d) = -10^{-d}`
    
    - If :math:`c(a,d) \\neq 0`:

        - If :math:`a = n`:
        
            - Rounding formula (if :math:`s(a,d) \\geq 5`): :math:`r(a,d) = \\frac{\\lceil a\\cdot{10^d} \\rceil}{10^d}`
            - Rounding formula (if :math:`s(a,d) < 5`): :math:`r(a,d) = \\frac{\\lfloor a\\cdot{10^d} \\rfloor}{10^d}`
        
        - If :math:`a \\neq n`:
        
            - Rounding formula (if :math:`s(a,d) \\geq 5`): :math:`r(a,d) = -\\frac{\\lceil a\\cdot{10^d} \\rceil}{10^d}`
            - Rounding formula (if :math:`s(a,d) < 5`): :math:`r(a,d) = -\\frac{\\lfloor a\\cdot{10^d} \\rfloor}{10^d}`
    
    - |rounding|

    Examples
    --------
    Import `rounded_value` function from `regressions` library
        >>> from regressions.statistics.rounding import rounded_value
    Round the number 9.2157823956916472 to six decimal places
        >>> number_normal = rounded_value(9.2157825956916472, 6)
        >>> print(number_normal)
        9.215783
    Round the number -0.00000003 to six decimal places
        >>> number_abnormal = rounded_value(-0.00000003, 6)
        >>> print(number_abnormal)
        -1e-6
    Round the number 11.725371548561 to four decimal places (without providing a value for the precision argument)
        >>> round_skip = rounded_value(11.725371548561)
        >>> print(round_skip)
        11.7254
    """
    # Handle input errors
    allow_none_scalar(number)
    positive_integer(precision)

    # Handle None value
    if number == None:
        return None
    
    # Circumvent rounding to zero with small positive numbers
    elif number < 10**(-precision) and number > 0:
        return 10**(-precision)

    # Circumvent rounding to zero with small negative numbers
    elif number > -10**(-precision) and number < 0:
        return -10**(-precision)
    
    # Handle general case
    else:
        return float(round(number, precision))

def rounded_list(numbers, precision = 4):
    # Handle input errors
    allow_none_vector(numbers, 'first')
    positive_integer(precision)

    # Create list to return
    results = []

    # Iterate over input
    for number in numbers:
        # Store rounded values of input in list to return
        results.append(rounded_value(number, precision))
    
    # Return results
    return results