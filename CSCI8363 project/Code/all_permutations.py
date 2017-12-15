import itertools

#function to find all permutations of a list for coefficients
def get_all_permutations(padded_addends):
    ununique_out_L = []
    unique_out_L = []
    for L in padded_addends:
        ununique_out_L += [list(perm) for perm in itertools.permutations(L)]
    for subL in ununique_out_L:
        if not subL in unique_out_L:
            unique_out_L += [subL]

    return unique_out_L

def list_transpose(L):
    return map(list, zip(*L))

def all_permutations(padded_addends):
    return list_transpose(get_all_permutations(padded_addends))



