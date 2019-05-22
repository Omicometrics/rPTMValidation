from bisect import bisect_left

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
def annotate(double[:] mzs, theor_ions, double tol=0.2):
    """
    Annotates the spectrum using the provided theoretical ions.

    Args:
        theor_ions (list): The list of theoretical Ions.
        tol (float, optional): The mass tolerance for annotations.

    Returns:
        A dictionary of ion label to Annotation namedtuple.

    """
    cdef int npeaks, idx, pos
    
    npeaks = mzs.shape[0]
    anns = {}
    for mass, label, pos in theor_ions:
        idx = bisect_left(mzs, mass)
        if idx > 0 and mass - mzs[idx - 1] <= tol:
            anns[label] = (idx - 1, mass - mzs[idx - 1], pos)
        elif idx < npeaks and mzs[idx] - mass <= tol:
            anns[label] = (idx, mass - mzs[idx], pos)

    return anns