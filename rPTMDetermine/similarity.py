#! /usr/bin/env python3
"""
A module for evaluating the similarity of two MS/MS spectra, based on their
ion annotations and intensities.

"""
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import tqdm
import collections

from .peptide_spectrum_match import PSM, SimilarityScore, UnmodPSM
from .psm_container import PSMContainer
from .mass_spectrum import Spectrum


def calculate_similarity_scores(mod_psms: PSMContainer[PSM],
                                unmod_psms: PSMContainer[UnmodPSM]) \
        -> PSMContainer[PSM]:
    """
    Calculates the similarity between the mass spectra of modified and
    unmodified peptides.

    Args:
        mod_psms (PSMContainer of PSMs): The modified PSMs.
        unmod_psms (PSMContainer of UnmodPSMs): The unmodified PSMs.

    Returns:
        The modified PSMs, with their similarity scores now set.

    """
    # Note that the index dictionary requires a tuple to be passed as the key
    index: Dict[Tuple[str, ...], List[int]] = mod_psms.get_index(("uid",))

    # do denoising to unmodified PSM in advance
    mod_index, unmod_spec_info = collections.defaultdict(list), []
    for i, upsm in enumerate(unmod_psms):
        ions, spec = upsm.denoise_spectrum()
        unmod_spec_info.append((upsm.data_id, upsm.spec_id, ions, spec))
        for psm_uid in upsm.get_mod_ids():
            j = index[(psm_uid,)][0]
            mod_index[j].append(i)
        upsm.peptide.clean_fragment_ions()

    # calculate similarities
    for i in tqdm.tqdm(mod_index.keys()):
        psm = mod_psms[i]
        mod_ions, mod_spec = psm.denoise_spectrum()
        scores = []
        for j in mod_index[i]:
            data_id, spec_id, unmod_ions, unmod_spec = unmod_spec_info[j]
            s = _calculate_spectral_similarity(
                mod_ions, mod_spec, unmod_ions, unmod_spec
            )
            scores.append(SimilarityScore(data_id, spec_id, s))
        psm.similarity_scores = scores

    mod_psms.clean_fragment_ions()

    return mod_psms


def calculate_spectral_similarity(psm1: PSM, psm2: PSM) -> float:
    """
    Calculates the similarity between two spectra based on their annotations
    and intensities. This function computes the dot product of the two
    spectra.

    Args:
        psm1 (peptide_spectrum_match.PSM): The first PSM, with an associated
                                           Spectrum.
        psm2 (peptide_spectrum_match.PSM): The second PSM, with an associated
                                           Spectrum.

    Returns:
        float: The dot product similarity of the spectra.

    """
    # Ensure that the spectra have been denoised and get the ion annotations
    ions1, spec1 = psm1.denoise_spectrum()
    ions2, spec2 = psm2.denoise_spectrum()
    return _calculate_spectral_similarity(ions1, spec1, ions2, spec2)


def _calculate_spectral_similarity(ions1, spec1, ions2, spec2,
                                   add_=np.add.reduce):
    """ Calculate spectra similarity """
    # Get the peak indices of the peaks which match between the two spectra
    idx1, idx2 = match_spectra((spec1, ions1), (spec2, ions2))

    # square rooted intensities
    sqrt_ints1, sqrt_ints2 = np.sqrt(spec1.intensity), np.sqrt(spec2.intensity)

    # Calculate the dot product of the spectral matches
    int_product = np.dot(sqrt_ints1[idx1], sqrt_ints2[idx2])

    sqrt_sum_ints1 = np.sqrt(add_(spec1.intensity))
    sqrt_sum_ints2 = np.sqrt(add_(spec2.intensity))

    return int_product / (sqrt_sum_ints1 * sqrt_sum_ints2)


def match_spectra(spectrum1: Tuple[Spectrum, dict],
                  spectrum2: Tuple[Spectrum, dict]) -> Tuple[int, int]:
    """
    Finds the indices of the matching peaks between two mass spectra.

    Args:
        spectrum1 (tuple): A tuple of (Spectrum, dict of ions)
        spectrum2 (tuple): A tuple of (Spectrum, dict of ions)

    Returns:
        list of two-tuples: The peak index in spectrum1 and the peak index in
                            spectrum2.

    """
    spec1, ions1 = spectrum1
    spec2, ions2 = spectrum2

    # bym contains the b, y and precursor ion names
    # neutrals contains the neutral losses from these ions
    # set of indices
    bym1, neutrals1, idx_set1 = _annotation_names(ions1)
    bym2, neutrals2, idx_set2 = _annotation_names(ions2)

    # Get the peak indices of the matched fragments
    matched_by1, matched_by2 = \
        _matched_peak_indices(ions1, ions2, bym1 & bym2)
    matched_neut1, matched_neut2 = \
        _matched_peak_indices(ions1, ions2, neutrals1 & neutrals2)

    # Remove replicate indices
    mindex1, mindex2 = _merged_matches(matched_by1, matched_by2, spec2)
    mset_idx1, mset_idx2 = set(mindex1), set(mindex2)

    # Find the non b, y or precursor fragments
    try:
        neut1, neut2 = zip(
            *[(ii, jj) for ii, jj in zip(matched_neut1, matched_neut2)
              if ii not in mset_idx1 and jj not in mset_idx2]
        )
    except ValueError:
        pass
    else:
        nix1, nix2 = _merged_matches(neut1, neut2, spec2)
        mindex1 += nix1
        mindex2 += nix2

    # m/z
    peak_mz1, peak_mz2 = spec1.mz, spec2.mz
    un_index1 = sorted(set(range(peak_mz1.size)) - idx_set1)
    un_index2 = np.array(
        sorted(set(range(peak_mz2.size)) - idx_set2), dtype=int
    )

    # Find the matched but unannotated ions
    if not un_index1 or un_index2.size == 0:
        return mindex1, mindex2

    diff_mz = np.absolute(peak_mz2[un_index2]
                          - peak_mz1[un_index1][:, np.newaxis])
    for idx1, diff in zip(un_index1, diff_mz):
        if (diff <= 0.2).any():
            peak_idx = un_index2[diff <= 0.2]
            if peak_idx.size == 1:
                idx2 = peak_idx[0]
            else:
                peak_ints = spec2.select(peak_idx, cols=1)
                idx2 = peak_idx[np.argmax(peak_ints)]
            mindex1.append(idx1)
            mindex2.append(idx2)

    return mindex1, mindex2


def _matched_peak_indices(ions1: Dict[str, Tuple[int, int]],
                          ions2: Dict[str, Tuple[int, int]],
                          common_ions: Iterable[str]):
    """
    Finds the index of the peak in the mass spectra for each of the ions
    commonly annotated between the spectra.

    Args:
        ions1 (dict): A dictionary mapping an ion label to a tuple of
                      (peak index, ion index in the peptide).
        ions2 (dict): A dictionary mapping an ion label to a tuple of
                      (peak index, ion index in the peptide).
        common_ions (set): The labels for the ions commonly annotated in
                           both spectra.

    Returns:
        tuple: (peak positions in first spectrum, peak positions in second
                spectrum.

    """
    if not common_ions:
        return [], []
    return zip(*[(ions1[ion][0], ions2[ion][0]) for ion in common_ions])


def _annotation_names(ions: Dict[str, Tuple[int, int]])\
        -> Tuple[Set[str], Set[str], Set[int]]:
    """
    Separates the annotations by name into two groups: y/b/M ions and
    others.

    Args:
        ions (dict): A dictionary of ion label to (peak index, peptide index).

    Returns:
        tuple of two sets: y/b/M ions and other ions.

    """
    frags, neu, index = set(), set(), set()
    for ion, (i, _) in ions.items():
        if ion[0] in "ybM" and "-" not in ion:
            frags.add(ion)
        else:
            neu.add(ion)
        index.add(i)
    return frags, neu, index


def _merged_matches(indices1: Sequence[int], indices2: Sequence[int],
                    spec2: Spectrum) -> Tuple[Any, Any]:
    """
    Merges the matched peak indices into a single list of tuples, removing
    replicates in the process.

    Args:
        indices1 (list): The peak indices of the first set of annotations.
        indices2 (list): The peak indices of the second set of annotations.
        spec2 (spectrum.Spectrum): The second mass spectrum.

    Returns:
        matched indices
    """
    matched_index1, matched_index2 = [], []
    if len(set(indices1)) < len(indices1):
        for idx1 in set(indices1):
            if indices1.count(idx1) == 1:
                idx2 = indices2[indices1.index(idx1)]
            else:
                # Find the duplicates and keep only the one with the highest
                # peak intensity
                m_indices2 = [indices2[ii] for ii, idx in enumerate(indices1)
                              if idx == idx1]
                peak_ints = list(spec2.select(m_indices2, cols=1))
                idx2 = m_indices2[peak_ints.index(max(peak_ints))]
            matched_index1.append(idx1)
            matched_index2.append(idx2)

        return matched_index1, matched_index2

    return list(indices1), list(indices2)
