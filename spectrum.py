#! /usr/bin/env python3
"""
This module provides functions for processing mass spectra.

"""
import bisect
import collections
import operator

import numpy as np

from constants import ITRAQ_MASSES


Annotation = collections.namedtuple("Annotation",
                                    ["peak_num", "mass_diff", "ion_pos"])


class Spectrum():
    """
    A class to represent a mass spectrum. The class composes a numpy array to
    store the spectral signals and provides methods for manipulating and
    exploring the mass spectrum.

    """
    def __init__(self, peak_list):
        """
        Initializes the class.

        Args:
            peak_list (list): A list of lists containing m/z, intensity pairs.

        """
        self._peaks = np.array(peak_list)
        # Sort the spectrum by the m/z ratios
        self._mz_sort()

    def __iter__(self):
        """
        Implements the __iter__ method for the Spectrum class, using the
        composed numpy array.

        """
        return self._peaks.__iter__()

    def __getitem__(self, indices):
        """
        Implements the __getitem__ method for the Spectrum class, using the
        composed numpy array.

        """
        return self._peaks[indices]

    def __len__(self):
        """
        Implements the __len__ method for the Spectrum class, using the
        composed numpy array.

        Returns:
            The length of the Spectrum object as an int.

        """
        return len(self._peaks)

    def __repr__(self):
        """
        Implements the __repr__ method for the Spectrum class, using the
        composed numpy array.

        Returns:
            The official string representation of the Spectrum object.

        """
        return repr(self._peaks)

    def __nonzero__(self):
        """
        Implements the __nonzero__ method for the Spectrum class, testing
        whether the underlying numpy array has been populated.

        """
        return self._peaks.size > 0

    def _mz_sort(self):
        """
        Sorts the spectrum by the m/z ratios.

        """
        self._peaks = self._peaks[self._peaks[:, 0].argsort()]

    @property
    def mz(self):
        """
        Retrieves the mass/charge ratios of the spectrum peaks.

        """
        return self._peaks[:, 0]

    @property
    def intensity(self):
        """
        Retrieves the intensities of the spectrum peaks.

        """
        return self._peaks[:, 1]

    def select(self, peaks, col=None):
        """
        Extracts only those peak indices in the given list.

        Args:
            peaks (list): A list of peak indices.
            cols (int, optional): The column(s) to retrieve. If None,
                                  retrieve all.

        Returns:
            Current Spectrum object filtered by the given indices.

        """
        return self._peaks[peaks, col if col is not None else ":"]

    def normalize(self):
        """
        Normalizes the spectrum to the base peak.

        Returns:
            Spectrum

        """
        self._peaks[:, 1] = self._peaks[:, 1] / self.max_intensity()
        return self

    def max_intensity(self):
        """
        Finds the maximum intensity in the spectrum.

        Returns:
            The maximum intensity as a float.

        """
        return self.intensity.max()

    def centroid(self):
        """
        Centroids a tandem mass spectrum according to the m/z differences.
        All fragment ions with adjacent m/z differences of less than 0.1 Da
        are centroided into the ion with the highest intensity.

        Returns:
            Centroided Spectrum object.

        """
        if len(self._peaks) <= 1:
            return self

        mz_diffs = np.diff(self.mz)

        centroided = []
        idx = 0
        while idx < len(self._peaks):
            peak = self._peaks[idx]
            if idx >= len(mz_diffs):
                centroided.append(peak)
                break
            diff = mz_diffs[idx]
            if diff > 0.1:
                centroided.append(peak)
            else:
                peak_cluster = [peak]
                df = 0
                while df <= 0.1:
                    idx += 1
                    peak = self._peaks[idx]
                    peak_cluster.append(peak)
                    if idx == len(mz_diffs):
                        break
                    df = mz_diffs[idx]
                if len({p[1] for p in peak_cluster}) == 1:
                    centroided.append(np.array(
                        [sum(p[0] for p in peak_cluster) /
                         float(len(peak_cluster)),
                         peak_cluster[0][1]]))
                else:
                    centroided.append(max(peak_cluster,
                                          key=operator.itemgetter(1)))
            idx += 1

        self._peaks = np.array(centroided)

        return self

    def remove_itraq(self, tol=0.1):
        """
        Removes the iTRAQ fragment peaks from the spectrum.

        Args:
            tol (float, optional): The mass tolerance.

        Returns:
            Spectrum object minus any iTRAQ peaks.

        """
        self._peaks = self._peaks[~np.isin(self._peaks[:, 0], ITRAQ_MASSES)]
        return self

    def annotate(self, theor_ions, tol=0.2):
        """
        Annotates the spectrum using the provided theoretical ions.

        Args:
            theor_ions (list): The list of theoretical Ions.
            tol (float, optional): The mass tolerance for annotations.

        Returns:
            A dictionary of ion label to Annotation namedtuple.

        """
        mz = list(self._peaks[:, 0])
        npeaks = len(self._peaks)
        insert_idxs = [bisect.bisect_left(mz, ion.mass) for ion in theor_ions]

        anns = {}
        for idx, (mass, label, pos) in zip(insert_idxs, theor_ions):
            if idx > 0 and mass - mz[idx - 1] <= tol:
                anns[label] = Annotation(idx - 1, mass - mz[idx - 1], pos)
            elif idx < npeaks and mz[idx] - mass <= tol:
                anns[label] = Annotation(idx, mass - mz[idx], pos)

        return anns

    def denoise(self, assigned_peaks, max_peaks_per_window=8):
        """
        Denoises the mass spectrum using the annotated ions.

        Args:
            assigned_peaks (list): A list of booleans indicating whether the
                                   corresponding index peak is annotated.
            max_peaks_per_window (int, optional): The maximum number of peaks
                                                  to include per 100 Da window.

        Returns:
            The denoised peak indexes as a list.

        """
        npeaks = len(self._peaks)
        # Divide the mass spectrum into windows of 100 Da
        n_windows = int((self._peaks[-1][0] - self._peaks[0][0]) / 100.) + 1
        start_idx, peaks = 0, []

        for window in range(n_windows):
            # Set up the mass limit for the current window
            max_mass = self._peaks[0][0] + (window + 1) * 100.

            # Find the last index with a peak mass within the current window
            for end_idx in range(start_idx, npeaks):
                if self._peaks[end_idx][0] > max_mass:
                    break

            if end_idx == start_idx:
                if (self._peaks[end_idx][0] <= max_mass and
                        assigned_peaks[end_idx]):
                    peaks.append(end_idx)
                continue

            # Sort the peaks within the window in descending order of
            # intensity
            window_peaks = sorted(list(range(start_idx, end_idx)),
                                  key=lambda ii: self._peaks[ii][1],
                                  reverse=True)

            ion_scores = [assigned_peaks[idx] for idx in window_peaks]

            # Sum the scores for the top intensity peaks in the window
            sum_scores = [sum(ion_scores[:idx])
                          for idx in range(1, min(len(ion_scores) + 1,
                                                  max_peaks_per_window + 1))]

            # Take the top number of peaks with the highest number of
            # annotations
            peaks += window_peaks[:sum_scores.index(max(sum_scores)) + 1]

            start_idx = end_idx

        return peaks
