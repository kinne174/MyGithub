from addend_finder import addend_finder
from padding import padding
from all_permutations import all_permutations
from coefficient_assigner import get_coefficients

import numpy as np
import math

#main function to generate half Gaussian kernel expansion
def feature_generator(in_arr, degree):
    print "Starting to generate half Gaussian feature vector"
    out_arr = np.array([math.exp(-1*sum(x*x for x in in_arr))], 'float')
    terms = len(in_arr)
    for d in range(1, degree+1):
        addends = addend_finder(d)
        addends.sort(key = len)
        padded_addends = padding(d,terms,addends)
        powers = np.array(all_permutations(padded_addends))
        coefficients = math.sqrt(float((2**d))/math.factorial(d))*np.array(get_coefficients(padded_addends))

        temp_arr = np.array([1]*len(powers[0]), 'float')
        for ind_elem in range(len(powers)):
            temp_arr *= in_arr[ind_elem]**powers[ind_elem]


        temp_arr *= coefficients
        temp_arr *= np.array([math.exp(-1*sum(x*x for x in in_arr))]*len(powers[0]), 'float')

        out_arr = np.append(out_arr, temp_arr)

    print "Ending generating of half Gaussian feature vector"

    return out_arr

def kernel_trick(in_arr_1, in_arr_2):
    for ind in range(len(in_arr_1)):
        difference_list = in_arr_1-in_arr_2
    return math.exp(-1*sum(x*x for x in difference_list))
        