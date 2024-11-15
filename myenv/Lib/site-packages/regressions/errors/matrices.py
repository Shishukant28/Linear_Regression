from .positions import argument_position

def confirm_matrix(matrix, position = 'only'):
    identifier = argument_position(position)
    if not isinstance(matrix, list) or not isinstance(matrix[0], list) or isinstance(matrix[0][0], list):
        raise TypeError(f'{identifier.capitalize()} must be a 2-dimensional list')
    else:
        return f'{identifier.capitalize()} is a 2-dimensional list'

def matrix_of_scalars(matrix, position = 'only'):
    confirm_matrix(matrix, position)
    identifier = argument_position(position)
    for vector in matrix:
        if not isinstance(vector, list):
            raise TypeError(f'Elements within {identifier} must be lists')
        for scalar in vector:
            if not isinstance(scalar, (int, float)):
                raise TypeError(f'Elements within lists within {identifier} must be integers or floats')
    else:
        return f'{identifier.capitalize()} is a 2-dimensional list containing nested elements that are integers or floats'

def square_matrix(matrix):
    matrix_of_scalars(matrix, 'first')
    if len(matrix) != len(matrix[0]):
        raise ValueError('First argument must contain the same amount of lists as the amount of elements contained within its first list')
    else:
        return 'First argument contains the same amount of lists as the amount of elements contained within its first list'

def compare_rows(matrix_one, matrix_two):
    matrix_of_scalars(matrix_one, 'first')
    matrix_of_scalars(matrix_two, 'second')
    if len(matrix_one) != len(matrix_two):
        raise ValueError('First argument and second argument must contain the same amount of lists')
    else:
        return 'First argument and second argument contain the same amount of lists'

def compare_columns(matrix_one, matrix_two):
    matrix_of_scalars(matrix_one, 'first')
    matrix_of_scalars(matrix_two, 'second')
    if len(matrix_one[0]) != len(matrix_two[0]):
        raise ValueError('First list within first argument and first list within second argument must contain the same amount of elements')
    else:
        return 'First list within first argument and first list within second argument contain the same amount of elements'

def compare_matrices(matrix_one, matrix_two):
    compare_rows(matrix_one, matrix_two)
    compare_columns(matrix_one, matrix_two)
    return 'First argument and second argument contain the same amount of lists; first list within first argument and first list within second argument contain the same amount of elements'

def columns_rows(matrix_one, matrix_two):
    matrix_of_scalars(matrix_one, 'first')
    matrix_of_scalars(matrix_two, 'second')
    if len(matrix_one[0]) != len(matrix_two):
        raise ValueError('First list within first argument must contain the same amount of elements as the amount of lists contained within second argument')
    else:
        return 'First list within first argument contains the same amount of elements as the amount of lists contained within second argument'

def allow_none_matrix(matrix, position = 'only'):
    identifier = argument_position(position)
    if not isinstance(matrix, list):
        raise TypeError(f'{identifier.capitalize()} must be a list')
    if not isinstance(matrix[0], list) and matrix[0] is not None:
        raise TypeError(f'{identifier.capitalize()} must contain either lists or None')
    if len(matrix) > 1 and not isinstance(matrix[1], list):
        raise TypeError(f'Second element of {identifier} must be a list')
    if len(matrix) > 1 and not isinstance(matrix[0][0], (int, float)):
        raise TypeError(f'If the first element of {identifier} is not None, then it must be a list of integers or floats')
    if len(matrix) > 1 and not isinstance(matrix[-1][0], str):
        raise TypeError(f'If the first element of {identifier} is not None, then the last element must be a list with a first element that is a string')
    else:
        return f'{identifier.capitalize()} is either a 1-dimensional list that contains None or a 2-dimensional list containing lists in which the first nested list contains integers or floats and the last nested list begins with a string'

def allow_vector_matrix(matrix, position = 'only'):
    identifier = argument_position(position)
    if not isinstance(matrix, list):
        raise TypeError(f'{identifier.capitalize()} must be a list')
    if not isinstance(matrix[0], (list, int, float, str)) and matrix[0] is not None:
        raise TypeError(f'Elements of {identifier} must lists, integers, floats, strings, or None')
    if len(matrix) > 1 and isinstance(matrix[0], list) and isinstance(matrix[0][0], list):
        raise TypeError(f'{identifier.capitalize()} cannot be more than a 2-dimensional list')
    else:
        return f'{identifier.capitalize()} is a list containing lists, integers, floats, strings, or None'

def level(matrix, scalar):
    if not scalar <= len(matrix[0]):
        raise ValueError('Last argument must be less than or equal to the length of the nested lists within the first argument')
    else:
        return 'Last argument is less than or equal to the length of the nested lists within the first argument'