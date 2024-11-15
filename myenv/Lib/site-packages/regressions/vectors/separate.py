from regressions.errors.vectors import allow_none_vector

def separate_elements(vector):
    """
    Separates a single vector into two vectors based their data types (numerical and non-numerical)

    Parameters
    ----------
    vector : list of int or float or str
        List containing elements of various types: integers, floats, strings, or None

    Raises
    ------
    TypeError
        Elements of argument must be integers, floats, strings, or None

    Returns
    -------
    separate_vectors['numerical'] : list of int or float
        List containing only the integer or float elements from the input vector; may be an empty list if input vector contained no integer or float elements
    separate_vectors['other'] : list of str
        List containing only the string or None elements from the input vector; may be an empty list if input vector contained no string elements and did not merely contain None

    See Also
    --------
    :func:`~regressions.vectors.generate_generate_elements`, :func:`~regressions.vectors.unify.unite_vectors`

    Notes
    -----
    - Set of all real numbers: :math:`\\mathbb{R}`
    - Set of all strings: :math:`\\mathbb{S}`
    - Set containing None: :math:`\\{ \\emptyset \\}`
    - Set of all real numbers, all strings, and None: :math:`\\mathbb{A} = \\{ \\mathbb{R}, \\mathbb{S}, \\emptyset \\}`
    - Set of mixed elements: :math:`M = \\{ m \\mid m \\in \\mathbb{A} \\}`
    - Set of numerical elements from mixed elements: :math:`N = \\{ n \\mid n \\in M, n \\in \\mathbb{R} \\}`
    - Set of non-numerical elements from mixed elements: :math:`O = \\{ o \\mid o \\in M, o \\in \\mathbb{S} \\cup \\emptyset \\}`

    Examples
    --------
    Import `separate_elements` function from `regressions` library
        >>> from regressions.vectors.separate import separate_elements
    Separate [1, 'two', 3, 'four'] into two vectors based on its data types
        >>> separate_vectors_mixed = separate_elements([1, 'two', 3, 'four'])
        >>> print(separate_vectors_mixed['numerical'])
        [1, 3]
        >>> print(separate_vectors_mixed['other'])
        ['two', 'four']
    Separate [None] into two vectors based on its data types
        >>> separate_vectors_none = separate_elements([None])
        >>> print(separate_vectors_none['numerical'])
        []
        >>> print(separate_vectors_none['other'])
        [None]
    """
    # Handle input errors
    allow_none_vector(vector)

    # Create lists to store separate values
    numerical_elements = []
    other_elements = []

    # Iterate over input
    for element in vector:
        # Store numerical elements
        if isinstance(element, (int, float)):
            numerical_elements.append(element)
        
        # Store non-numerical elements
        else:
            other_elements.append(element)
    
    # Package both types of element in single dictionary to return
    results = {
        'numerical': numerical_elements,
        'other': other_elements
    }
    return results