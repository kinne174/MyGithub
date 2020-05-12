def addend_finder(num):
    return addend_finder_wrapper(num, num , [], [])

#function to return all addends of a positive integer
#credit given at bottom
def addend_finder_wrapper(n, max, L, out_list):
    if (n == 0):
        out_list = out_list + [L]
        return out_list
    for i in reversed(range(1,min(max,n) + 1)):
        out_list = addend_finder_wrapper(n - i, i, L + [i], out_list)
    return out_list

#credit to https://stackoverflow.com/questions/7331093/getting-all-possible-sums-that-add-up-to-a-given-number
# which leads to https://introcs.cs.princeton.edu/java/23recursion/Partition.java.html



