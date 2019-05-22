import cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport qsort

@cython.boundscheck(False)
def denoise(double[:, :] peaks, assigned_peaks, int max_peaks_per_window):
    """
    Denoises the mass spectrum using the annotated ions.

    Args:
        assigned_peaks (list): A list of booleans indicating whether the
                               corresponding index peak is annotated.
        max_peaks_per_window (int, optional): The maximum number of peaks
                                              to include per 100 Da window.

    Returns:
        Tuple: The denoised peak indexes as a list, The denoised spectrum

    """
    cdef ssize_t npeaks
    cdef int n_windows, start_idx, window, end_idx, idx, ii
    cdef double max_mass
    cdef int[:] window_peaks, ion_scores, sum_scores
    
    npeaks = peaks.shape[0]
    # Divide the mass spectrum into windows of 100 Da
    n_windows = int((peaks[-1][0] - peaks[0][0]) / 100.) + 1
    start_idx, new_peaks = 0, []

    for window in range(n_windows):
        # Set up the mass limit for the current window
        max_mass = peaks[0][0] + (window + 1) * 100.

        # Find the last index with a peak mass within the current window
        for end_idx in range(start_idx, npeaks):
            if peaks[end_idx][0] > max_mass:
                break

        if end_idx == start_idx:
            if (peaks[end_idx][0] <= max_mass and
                    assigned_peaks[end_idx]):
                new_peaks.append(end_idx)
            continue

        # Sort the peaks within the window in descending order of
        # intensity
        window_peaks = sorted(list(range(start_idx, end_idx)),
                              key=lambda ii: peaks[ii][1],
                              reverse=True)

        ion_scores = [assigned_peaks[idx] for idx in window_peaks]

        # Sum the scores for the top intensity peaks in the window
        sum_scores = [sum(ion_scores[:idx])
                      for idx in range(1, min(len(ion_scores) + 1,
                                              max_peaks_per_window + 1))]

        # Take the top number of peaks with the highest number of
        # annotations
        new_peaks += window_peaks[:sum_scores.index(max(sum_scores)) + 1]

        start_idx = end_idx

    return new_peaks