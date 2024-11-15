from regressions.statistics.rounding import rounded_value

def no_zeroes(coefficients, precision = 4):
    result = []
    for coefficient in coefficients:
        if coefficient == 0:
            result.append(10**(-precision))
        else:
            result.append(rounded_value(coefficient, precision))
    return result