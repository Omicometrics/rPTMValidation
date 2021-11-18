#! /usr/bin/env python3
"""
A module for evaluating the similarity of two MS/MS spectra, based on their
ion annotations and intensities.

"""
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple, Optional

import numpy as np
import tqdm
import collections

from operator import itemgetter

from pepfrag import constants

from .base import SimilarityScore
from .peptide_spectrum_match import PSM
from .mass_spectrum import Spectrum
from .peptides import merge_seq_mods


MH = constants.FIXED_MASSES["H"]
MH2O = constants.FIXED_MASSES["H2O"]
MNH3 = constants.FIXED_MASSES["NH3"]


def _remove_precursors(ions: Dict[str, Tuple[int, int]],
                       spectrum: Spectrum, charge: int, tol: float = 0.2)\
        -> Optional[Tuple[Dict[str, int], Spectrum]]:
    """
    Remove precursor and neutral loss of precursor peaks,
    with those around 1Da for isotopic peaks.
    """
    mz = spectrum.mz
    pmass = (spectrum.prec_mz - MH) * charge
    # precursor m/z with neutral losses should be removed
    retain_peaks = np.ones(mz.size, dtype=bool)
    for c in range(1, charge + 1):
        pmz_series = [pmass / charge + MH,
                      (pmass - MH2O) / charge + MH,
                      (pmass - MNH3) / charge + MH]
        for pmz in pmz_series:
            retain_peaks[np.absolute(mz - pmz) <= tol] = False

    # remove precursor related peaks
    if not retain_peaks.any():
        return None, None

    # ions
    ions = {ion: j for ion, (j, k) in ions.items()}
    if not retain_peaks.all():
        # if the peaks are annotated by other types of theoretical ions,
        # keeps it
        non_prec_idx = [j for ion, (j, k) in ions.items()
                        if "M" not in ion and not retain_peaks[j]]
        if non_prec_idx:
            retain_peaks[non_prec_idx] = True
        peaks = spectrum._peaks[retain_peaks, :]
        # remove precursors in the ion names and update the index
        retain_idx, = np.where(retain_peaks)
        ix2 = {j: i for i, j in enumerate(retain_idx)}
        ions = {ion: ix2[j] for ion, j in ions.items() if "M" not in ion}

        return ions, Spectrum(peaks, spectrum.prec_mz, None)

    return ions, spectrum


def _normalize_peak_intensity(spectrum: Spectrum) -> Spectrum:
    """ Normalize intensities of peaks for similarity score calculation.

    """
    spectrum[:1] = np.sqrt(spectrum[:, 1]) / np.sqrt(spectrum[:, 1].sum())
    return spectrum


def _unique_match(index: Sequence[int], intensity: np.ndarray,
                  ref_index: Sequence[int], ref_intensity: np.ndarray)\
        -> Tuple[np.ndarray, np.ndarray]:
    """ Unique matches between peaks in both mass spectra.

    """
    def _get_unique_index(index1, index2, intensity2):
        """ Gets unique index. """
        if len(set(index1)) == len(index1):
            return index1, index2
        # unique index
        idx = collections.defaultdict(list)
        for j, i in enumerate(index1):
            idx[i].append(index2[j])
        # retain the one with the highest intensities
        ix1, ix2 = sorted(idx.keys()), []
        for i in ix1:
            ixk = idx[i]
            ix2.append(ixk[np.argmax(intensity2[ixk])]
                       if len(ixk) > 1 else ixk[0])
        return ix1, ix2

    idx1, idx2 = _get_unique_index(index, ref_index, ref_intensity)
    idx2, idx1 = _get_unique_index(idx2, idx1, intensity)

    return intensity[idx1], ref_intensity[idx2]


def _calculate_spectral_similarities(target_spectrum, ref_spectra):
    """ Reimplementation of calculate spectra similarities """
    ions, spec = target_spectrum
    spec_int = spec.intensity
    ion_set = set(ions.keys())

    # get common ions with their numbers
    common_ions = []
    for i, (uid, ions_ref, spec_ref) in enumerate(ref_spectra):
        tmp_ions = ion_set & set(ions_ref.keys())
        common_ions.append((len(tmp_ions), i, tmp_ions))

    # retain the top 50 numbers of common ions
    common_ions.sort(key=itemgetter(0), reverse=True)

    # similarities
    s = []
    for _, i, common_ion in common_ions[:50]:
        uid, ions_ref, spec_ref = ref_spectra[i]
        # Get the peak indices of the peaks which match between the two spectra
        idx = [ions[ion] for ion in common_ion]
        idx_ref = [ions_ref[ion] for ion in common_ion]

        # unique index
        if len(set(idx)) < len(idx) or len(set(idx_ref)) < len(idx_ref):
            intens1, intens2 =\
                _unique_match(idx, spec_int, idx_ref, spec_ref.intensity)
        else:
            intens1, intens2 = spec_int[idx], spec_ref.intensity[idx_ref]

        # Calculate the dot product of the spectral matches
        sk = (intens1 * intens2).sum()
        if sk > s[1]:
            s.append((uid, s))

    return s


def calculate_similarity_scores(mod_psms: Sequence[PSM],
                                unmod_psms: Sequence[PSM],
                                except_mods: Sequence[Tuple[str, str]] = None)\
        -> List[PSM]:
    """
    Calculates the similarity between the mass spectra.

    Args:
        mod_psms (List of PSMs): Target PSMs to calculate similarity
            scores.
        unmod_psms (List of Unmodified PSMs): Unmodified PSMs.
        except_mods (Set of (mod, site)): Modification exceptions to
            define unmodified peptides. Default is None, which indicates
            that no modification is exceptional. If the modification
            locates at peptide terminus, use `nterm` for N-terminus,
            `cterm` for C-terminus.

    Returns:
        The PSMs, with their similarity scores now set.

    """
    if except_mods is None:
        except_mods = set()

    # preprocess unmodified analogues to avoid repeating denoising
    unmod_specs: Dict[str, Tuple[str, Dict[str, int], Spectrum]] =\
        collections.defaultdict()
    for p in tqdm.tqdm(unmod_psms, desc="Process unmodified PSM spectra"):
        # denoising
        ions, spec = p.denoise_spectrum()
        # removes precursors
        ions, spec = _remove_precursors(ions, spec, p.charge)
        if ions is not None:
            # normalizes peak intensity
            spec = _normalize_peak_intensity(spec)
            # constructs the library in dictionary
            mpep = merge_seq_mods(p.seq, p.mods)
            unmod_specs[f"{mpep}#{p.charge}"] = (p.uid, ions, spec)

    # calculate similarities
    mod_specs = collections.defaultdict(list)
    for p in tqdm.tqdm(
            mod_psms,
            desc=("Calculate similarity scores"
                  " between modified and unmodified PSMs")
    ):
        # denoising
        mod_ions, mod_spec = p.denoise_spectrum()
        # removes precursors
        mod_ions, mod_spec = _remove_precursors(mod_ions, mod_spec, p.charge)
        if mod_ions is None:
            continue
        # normalizes peak intensity
        mod_spec = _normalize_peak_intensity(mod_spec)
        # calculate scores
        unmods = [m for m in p.mods if (m.mod, m.site if isinstance(m.site, str)
                                        else p.seq[m.site-1])
                  not in except_mods]
        unmod_pep = f"{merge_seq_mods(p.seq, unmods)}#{p.charge}"
        if unmod_pep in unmod_specs:
            match_scores = _calculate_spectral_similarities(
                (mod_ions, mod_spec), unmod_specs[unmod_pep]
            )
            p.similarity_scores = sorted(match_scores,
                                         key=itemgetter(1), reverse=True)
    return mod_psms
