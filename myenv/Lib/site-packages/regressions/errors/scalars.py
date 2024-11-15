from .positions import argument_position

def scalar_value(scalar, position = 'only'):
    identifier = argument_position(position)
    if not isinstance(scalar, (int, float)):
        raise TypeError(f'{identifier.capitalize()} must be an integer or a float')
    else:
        return f'{identifier.capitalize()} is an integer or a float'

def two_scalars(scalar_one, scalar_two):
    scalar_value(scalar_one, 'first')
    scalar_value(scalar_two, 'second')
    return 'Both first and second arguments are integers or floats'

def three_scalars(scalar_one, scalar_two, scalar_three):
    two_scalars(scalar_one, scalar_two)
    scalar_value(scalar_three, 'third')
    return 'First, second, and third arguments are all integers or floats'

def four_scalars(scalar_one, scalar_two, scalar_three, scalar_four):
    three_scalars(scalar_one, scalar_two, scalar_three)
    scalar_value(scalar_four, 'fourth')
    return 'First, second, third, and fourth arguments are all integers or floats'

def five_scalars(scalar_one, scalar_two, scalar_three, scalar_four, scalar_five):
    four_scalars(scalar_one, scalar_two, scalar_three, scalar_four)
    scalar_value(scalar_five, 'fifth')
    return 'First, second, third, fourth, and fifth arguments are all integers or floats'

def compare_scalars(scalar_one, scalar_two, position_one, position_two):
    scalar_value(scalar_one, position_one)
    scalar_value(scalar_two, position_two)
    if scalar_one > scalar_two:
        raise ValueError(f'{position_one.capitalize()} argument must be less than or equal to {position_two} argument')
    else:
        return f'{position_one.capitalize()} argument is less than or equal to {position_two} argument'

def positive_integer(scalar):
    if not isinstance(scalar, int) or not scalar > 0:
        raise ValueError('Last argument must be a positive integer')
    else:
        return 'Last argument is a positive integer'

def whole_number(scalar, position = 'only'):
    identifier = argument_position(position)
    if not isinstance(scalar, int) or not scalar >= 0:
        raise ValueError(f'{identifier.capitalize()} must be a whole number')
    else:
        return f'{identifier.capitalize()} is a whole number'

def select_integers(scalar, choices, position = 'only'):
    identifier = argument_position(position)
    if scalar not in choices:
        raise ValueError(f'{identifier.capitalize()} must be one of the following integers: {choices}')
    else:
        return f'{identifier.capitalize()} is one of the following integers: {choices}'

def allow_none_scalar(scalar, position = 'only'):
    identifier = argument_position(position)
    if not isinstance(scalar, (int, float)) and scalar is not None:
        raise TypeError(f'{identifier.capitalize()} must be an integer, a float, or None')
    else:
        return f'{identifier.capitalize()} is an integer, a float, or None'