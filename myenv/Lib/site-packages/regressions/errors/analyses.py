from .positions import argument_position

def select_equations(string):
    choices = ['linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', 'sinusoidal']
    if string not in choices:
        raise ValueError("First argument must be either 'linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', or 'sinusoidal'")
    else:
        return "First argument is either 'linear', 'quadratic', 'cubic', 'hyperbolic', 'exponential', 'logarithmic', 'logistic', or 'sinusoidal'"

def select_points(string = 'point', position = 'only'):
    identifier = argument_position(position)
    choices = ['point', 'intercepts', 'maxima', 'minima', 'inflections']
    if string not in choices:
        raise ValueError(f"{identifier.capitalize()} must be either 'point', 'intercepts', 'maxima', 'minima', or 'inflections'")
    else:
        return f"{identifier.capitalize()} is either 'point', 'intercepts', 'maxima', 'minima', or 'inflections'"