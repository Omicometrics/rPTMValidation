import cython

from binomial cimport _log_binom_prob

@cython.cdivision(True)
cdef double _ionscore(int seqlen, int npeaks, int seqcov, double mzrange, double tol):
    """
    Calculate the ion score for the PSM.

    """
    cdef int nbins
    cdef double mp
    
    # ion score
    nbins = int((mzrange / tol) + 0.5)
    mp = 0.
    if nbins > 0:
        mp = _log_binom_prob(seqcov, 2 * (seqlen - 1), npeaks / float(nbins))

    return mp / (seqlen ** 0.5)
    
cpdef double ionscore(int seqlen, int npeaks, int seqcov, double mzrange, double tol):
    return _ionscore(seqlen, npeaks, seqcov, mzrange, tol)