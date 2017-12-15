#function to determine if coefficients need zeroes or need to be cut down
def padding(degree, terms, addends):
    if degree > terms: #for example: (x1 + x2)^3
        addends = addends[:(max(ind for ind in range(len(addends)) if len(addends[ind]) == terms) + 1)]
        for a in range(len(addends)):
            cur_addend_length = len(addends[a])
            #choose_amount = misc.comb(terms,a+1)
            num_zeroes_needed = terms - cur_addend_length
            new_addend = addends[a] + [0]*num_zeroes_needed
            addends[a] = new_addend

    elif terms >= degree: #for example (x1 + x2 + x3)^2 or (x1 + x2)^2
        for a in range(len(addends)):
            cur_addend_length = len(addends[a])
            #choose_amount = misc.comb(terms,a+1)
            num_zeroes_needed = terms - cur_addend_length
            new_addend = addends[a] + [0]*num_zeroes_needed
            addends[a] = new_addend
    return addends



