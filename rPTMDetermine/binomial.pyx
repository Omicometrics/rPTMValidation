# distutils: language = c++

from libc.math cimport log10
from libcpp.vector cimport vector as cpp_vector


cdef double cursum
cdef int ii, jj
cdef int size = 500
cdef double LOGN[500]
cdef double CUMLOGN[500]

for ii in range(size):
    LOGN[ii] = log10(ii + 1)
    cursum = 0
    for jj in range(ii + 1):
        cursum += LOGN[jj]
    CUMLOGN[ii] = cursum


cdef double _log_binom_prob(int k, int n, double p):
    """
    Calculates the negative base-10 log of the cumulative binomial
    probability.

    Args:
        k (int): The number of successes.
        n (int): The number of trials.
        p (float): The probability of success.

    Returns:
        The negative base-10 log of the binomial CDF.

    """
    cdef int nk, i
    cdef double s, s1, s2, pbt, pbf, m, x, news
    cdef cpp_vector[double] pbk
    
    nk = n - k
    s = CUMLOGN[n - 1]

    pbt, pbf = log10(p), log10(1. - p)
    # the initial binomial
    s1, s2 = CUMLOGN[k - 1], CUMLOGN[nk - 1]
    pbk = cpp_vector[double]()
    m = s - s1 - s2 + k * pbt + nk * pbf
    pbk.push_back(m)
    # calculate the cumulative using recursive iteration
    for i in range(k, n):
        s1 += LOGN[i]
        s2 -= LOGN[nk - 1]
        nk -= 1
        news = s - s1 - s2 + (i + 1) * pbt + nk * pbf
        pbk.push_back(news)
        if news > m:
            m = news
    
    res = 0.
    for x in pbk:
        res += 10 ** (x - m)
        
    return - m - log10(res)


cpdef log_binom_prob(int k, int n, double p):
    return _log_binom_prob(k, n, p)
    