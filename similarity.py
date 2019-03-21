#! /usr/bin/env python3
"""
A module for evaluating the similarity of two MS/MS spectra, based on their
ion annotations and intensities.

"""
import collections
import os

import numpy as np

import peptides
from psm import PSM
import readers


def calculate_similarity_score(psm, pp_res, target_mod, data_sets):
    """
    Calculates the similarity score between the PSM spectrum and the
    unmodified analogues in the database search results.

    Args:
        psm (psm.PSM): The modified PSM.
        pp_res (dict): The ProteinPilot search results.
        target_mod (str): The name of the target modification.
        data_sets (dict): The config 'data_sets' dictionary.

    Returns:
        psm.PSM: The input PSM with the similarity_scores property updated.

    """
    # Remove the target modification from the PSM mods
    mods = [ms for ms in psm.mods if ms.mod != target_mod]
    peptide_str = peptides.merge_seq_mods(psm.sequence, mods)

    # Find the data_id/spec_id combinations which match the given PSM peptide
    match_ids = collections.defaultdict(list)
    for data_id, data in pp_res.items():
        for spec_id, matches in data.items():
            if test_matches_equal(matches, psm, peptide_str):
                match_ids[data_id].append(spec_id)

    similarities = []
    for data_id, spec_ids in match_ids.items():
        spec_file = os.path.join(data_sets[data_id]["data_dir"],
                                 data_sets[data_id]["spectra_file"])

        spectra = readers.read_spectra_file(spec_file)

        for spec_id in spec_ids:
            spec = spectra[spec_id].centroid().remove_itraq()

            # Calculate the similarity between the unmodified and modified
            # spectra
            unmod_psm = PSM(data_id, spec_id, psm.sequence, mods, psm.charge,
                            spectrum=spec)
            similarities.append(
                (data_id, spec_id,
                 calculate_spectral_similarity(psm, unmod_psm)))

    psm.similarity_scores = similarities

    return psm


def test_matches_equal(matches, psm, peptide_str):
    """
    Evaluates whether any one of a SpecMatch is to the same peptide,
    in terms of sequence and modifications, as the given PSM.

    Args:
        matches (list of SpecMatch): The matches to test.
        psm (psm.PSM): The peptide against which to compare.
        peptide_str (str): The peptide string including modifications.

    Returns:
        boolean: True if there is a match, False otherwise.

    """
    for match in matches:
        if match.seq != psm.sequence and match.theor_z != psm.charge:
            continue

        mods = match.mods

        if "Deamidated" in mods:
            # Remove deamidation from the mods string
            mods = ";".join(mod for mod in mods.split(";")
                            if not mod.startswith("Deamidated"))

        if peptides.merge_seq_mods(match.seq, mods) == peptide_str:
            return True

    return False


def calculate_spectral_similarity(psm1, psm2):
    """
    Calculates the similarity between two spectra based on their annotations
    and intensities. This function computes the dot product of the two
    spectra.

    Args:
        psm1 (psm.PSM): The first PSM, with an associated Spectrum.
        psm2 (psm.PSM): The second PSM, with an associated Spectrum.

    Returns:
        float: The dot product similarity of the spectra.

    """
    # Ensure that the spectra have been denoised
    ions1 = psm1.denoise_spectrum()
    ions2 = psm2.denoise_spectrum()

    spec1, spec2 = psm1.spectrum, psm2.spectrum

    # Get the peak indices of the peaks which match between the two spectra
    matched_idxs = match_spectra((spec1, ions1), (spec2, ions2))

    # Calculate the dot product of the spectral matches
    int_product = sum(np.sqrt(spec1[ii][1]) * np.sqrt(spec2[jj][1])
                      for ii, jj in matched_idxs
                      if ii is not None and jj is not None)

    sqrt_sum_ints1 = \
        np.sqrt(sum(spec1[ii][1] for ii, _ in matched_idxs if ii is not None))
    sqrt_sum_ints2 = \
        np.sqrt(sum(spec2[ii][1] for _, ii in matched_idxs if ii is not None))

    return int_product / (sqrt_sum_ints1 * sqrt_sum_ints2)


def match_spectra(spectrum1, spectrum2):
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
    bym1, neutrals1 = _annotation_names(ions1)
    bym2, neutrals2 = _annotation_names(ions2)

    # Get the peak indices of the matched fragments
    matched_by1, matched_by2 = \
        _matched_peak_indices(ions1, ions2, bym1 & bym2)
    matched_neut1, matched_neut2 = \
        _matched_peak_indices(ions1, ions2, neutrals1 & neutrals2)

    # Remove replicate indices
    matched_by = _merged_matches(matched_by1, matched_by2, spec2)
    idx_set1 = {ii for ii, _ in matched_by}
    idx_set2 = {ii for _, ii in matched_by}

    # Find the non b, y or precursor fragments
    neut1, neut2 = \
        zip(*[(ii, jj) for ii, jj in zip(matched_neut1, matched_neut2)
              if ii not in idx_set1 and jj not in idx_set2])

    matched_neut = _merged_matches(neut1, neut2, spec2)
    idx_set1.update(ii for ii, _ in matched_neut)
    idx_set2.update(ii for _, ii in matched_neut)

    # Combine the common indices
    matched_idxs = matched_by + matched_neut

    # Unmatched annotated fragments are left unmatched
    matched_idxs += [(idx, None) for idx, _ in ions1.values()
                     if idx not in idx_set1]
    matched_idxs += [(None, idx) for idx, _ in ions2.values()
                     if idx not in idx_set2]
    idx_set1.update(ii for ii, _ in ions1.values())
    idx_set2.update(ii for ii, _ in ions2.values())

    # Find the matched but unannotated ions
    for idx1, peak1 in enumerate(spec1):
        if idx1 in idx_set1:
            continue

        idx_set1.add(idx1)

        peak_idxs = \
            [idx2 for idx2, peak2 in enumerate(spec2) if idx2 not in idx_set2
             and abs(peak2[0] - peak1[0]) <= 0.2]

        if not peak_idxs:
            matched_idxs.append((idx1, None))
            continue

        if len(peak_idxs) == 1:
            idx2 = peak_idxs[0]
        else:
            peak_ints = spec2.select(peak_idxs, col=1)
            idx2 = peak_idxs[np.argmax(peak_ints)]

        matched_idxs.append((idx1, idx2))
        idx_set2.add(idx2)

    matched_idxs += [(None, idx) for idx in range(len(spec2))
                     if idx not in idx_set2]

    return matched_idxs


def _matched_peak_indices(ions1, ions2, common_ions):
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
    return zip(*[(ions1[ion][0], ions2[ion][0]) for ion in common_ions])


def _annotation_names(ions):
    """
    Separates the annotations by name into two groups: y/b/M ions and
    others.

    Args:
        ions (dict): A dictionary of ion label to (peak index, peptide index).

    Returns:
        tuple of two sets: y/b/M ions and other ions.

    """
    ions = set(ions.keys())
    frags = {l for l in ions if l[0] in "ybM" and "-" not in l}
    return frags, ions - frags


def _merged_matches(indices1, indices2, spec2):
    """
    Merges the matched peak indices into a single list of tuples, removing
    replicates in the process.

    Args:
        indices1 (list): The peak indices of the first set of annotations.
        indices2 (list): The peak indices of the second set of annotations.
        spec2 (spectrum.Spectrum): The second mass spectrum.

    Returns:
    """
    merged = []
    if len(set(indices1)) < len(indices1):
        for idx1 in set(indices1):
            if indices1.count(idx1) == 1:
                merged.append((idx1, indices2[indices1.index(idx1)]))
            else:
                # Find the duplicates and keep only the one with the highest
                # peak intensity
                m_indices2 = [indices2[ii] for ii, idx in enumerate(indices1)
                              if idx == idx1]
                peak_ints = list(spec2.select(m_indices2, col=1))
                merged.append(
                    (idx1, m_indices2[peak_ints.index(max(peak_ints))]))
    else:
        merged = list(zip(indices1, indices2))

    return merged
