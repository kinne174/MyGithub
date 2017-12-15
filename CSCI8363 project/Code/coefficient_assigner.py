import math
from operator import mul

#function to determine what unique coefficients are needed in expansion
def get_coefficients(padded_addends):
    terms = len(padded_addends[0])
    degree = padded_addends[0][0]
    outL = []

    for subL in padded_addends:
        unique_elements = []
        for elem in subL:
            if not elem in unique_elements:
                unique_elements += [elem]
        length_coefficients = math.factorial(terms)/reduce(mul,[math.factorial(subL.count(elem)) for elem in unique_elements],1)

        coefficient = math.sqrt(math.factorial(degree)/reduce(mul,[math.factorial(elem) for elem in subL]))

        outL += [coefficient]*length_coefficients

    return outL
