"""
This module contains functions related to PTM localization and isobaric
correction.

"""
import copy
import itertools
from typing import Container, List, Optional

import numpy as np

from pepfrag import ModSite, Peptide

from . import (
    machinelearning,
    PSM,
    PSMContainer
)
from .readers import PTMDB


def generate_localization_candidates(
        psm: PSM,
        mod: str,
        mod_mass: float,
        target_residues: Container[str]
) -> PSMContainer[PSM]:
    """
    Generates alternative localization candidates for `mod` in `psm`.

    Args:
        psm: The peptide spectrum match.
        mod: Target modification to be permuted.
        mod_mass: The mass of `mod`.
        target_residues: The amino acid residues targeted by `mod`.

    Returns:
        PSM candidates.

    """
    candidates: PSMContainer[PSM] = PSMContainer()

    mod_count = sum(ms.mod == mod for ms in psm.mods)
    if mod_count == 0:
        return PSMContainer()

    target_res_indices = [
        ii + 1 for ii, res in enumerate(psm.seq) if res in target_residues
    ]
    fixed_mods = [ms for ms in psm.mods if ms.mod != mod]

    temp_psm = PSM(
        psm.data_id,
        psm.spec_id,
        copy.deepcopy(psm.peptide),
        spectrum=psm.spectrum
    )
    temp_psm.mods = fixed_mods
    candidates.append(copy.deepcopy(temp_psm))

    for num_mods in range(1, mod_count + 1):
        site_combinations = itertools.combinations(
            target_res_indices, num_mods
        )
        for sites in site_combinations:
            mods = fixed_mods + \
                   [ModSite(mod_mass, j, mod) for j in sites]
            temp_psm.mods = mods
            candidates.append(copy.deepcopy(temp_psm))

    return candidates


def generate_alternative_nterm_candidates(
        psm: PSM,
        validation_mod: str,
        alternative_mod: str,
        mod_mass: float
) -> PSMContainer[PSM]:
    """
    Generates a candidate PSM with `alternative_mod` at the N-terminus.

    Args:
        psm: The peptide spectrum match.
        validation_mod: The modification under validation.
        alternative_mod: The alternative modification to consider at the
                         N-terminus.
        mod_mass: The mass of `mod`.

    Returns:
        PSM candidates.

    """
    candidates: PSMContainer[PSM] = PSMContainer()

    fixed_mods = [ms for ms in psm.mods if ms.mod != validation_mod]

    if not any(ms.site == 'nterm' for ms in psm.mods):
        mods = fixed_mods + [ModSite(mod_mass, 'nterm', alternative_mod)]

        temp_psm = PSM(
            psm.data_id,
            psm.spec_id,
            Peptide(psm.seq, psm.charge, mods),
            spectrum=psm.spectrum
        )
        candidates.append(temp_psm)

    return candidates


def generate_deamidation_candidates(
        psm: PSM,
        ptmdb: PTMDB
) -> PSMContainer[PSM]:
    """
    Generates alternative deamidation candidates for the given `psm`.

    Args:
        psm: The peptide spectrum match.
        ptmdb: The unimod modification database.

    Returns:
        PSM candidates.

    """
    return generate_localization_candidates(
        psm,
        'Deamidated',
        ptmdb.get_mass('Deamidated'),
        {'N', 'Q'}
    )


def generate_localization_isoforms(
        psm: PSM,
        mod: str,
        mod_mass: float,
        target_residue: str
) -> PSMContainer[PSM]:
    """
    Generates all possible isoforms of `mod` for the `psm`.

    The difference between this and generate_localization_candidates is that the
    latter will allow fewer instances of `mod` than originally exist in the
    `psm` modifications, whereas this implementation will not. This function
    may also only be applied to localize `mod` across a single `target_residue`.

    Args:
        psm: The peptide spctrum match.
        mod: Target modification to be permuted.
        mod_mass: The mass of `mod`.
        target_residue: The amino acid residue targeted by `mod`.

    Returns:
        PSM localization isoforms.

    """
    isoforms: PSMContainer[PSM] = PSMContainer()

    mod_sites = [
        ms.site for ms in psm.mods
        if ms.mod == mod and isinstance(ms.site, int) and
        psm.seq[ms.site - 1] == target_residue
    ]

    mod_count = len(mod_sites)

    possible_sites = [
        ii + 1 for ii, res in enumerate(psm.seq) if res == target_residue
    ]

    if mod_count == len(possible_sites):
        return PSMContainer([psm])

    fixed_mods = [
        ms for ms in psm.mods if ms.mod != mod and not
        (isinstance(ms.site, int) and psm.seq[ms.site - 1] == target_residue)
    ]

    temp_psm = PSM(
        psm.data_id,
        psm.spec_id,
        copy.deepcopy(psm.peptide),
        spectrum=copy.deepcopy(psm.spectrum)
    )

    for sites in itertools.combinations(possible_sites, mod_count):
        mods = fixed_mods + [ModSite(mod_mass, site, mod) for site in sites]

        temp_psm.mods = mods

        isoforms.append(copy.deepcopy(temp_psm))

    return isoforms


def localize(
        psm: PSM,
        target_mod: str,
        mod_mass: float,
        target_residue: str,
        model: machinelearning.Classifier,
        features: Optional[List[str]] = None
):
    """
    Localizes the `target_mod` in `psm` inplace.

    Args:
        psm: The peptide spectrum match.
        target_mod: The modification to localize.
        mod_mass: The mass of `target_mod`.
        target_residue: The amino acid residue targeted by `target_mod`.
        model: The trained PSM classifier.
        features: An optional subset of features to use in prediction.

    """
    isoforms = generate_localization_isoforms(
        psm,
        target_mod,
        mod_mass,
        target_residue
    )

    if len(isoforms) == 1:
        psm.site_prob = 1.
        psm.site_diff_score = 1.
        return psm

    for isoform in isoforms:
        isoform.extract_features()

    feature_array = isoforms.to_feature_array(features=features)
    scores = model.predict(feature_array)

    sorted_score_indices = scores.argsort()[::-1]

    # Find properties of the highest-scoring isoform
    max_score_idx = sorted_score_indices[0]
    max_score = scores[max_score_idx]
    site_prob = 1. / np.exp(scores - max_score).sum()
    diff_score = min(
        abs(max_score - scores[sorted_score_indices[1]] / max_score),
        1.
    )

    # Re-assign attributes of the highest-scoring isoform to the input PSM for
    # inplace update
    psm.mods = isoforms[max_score_idx].mods
    psm.features = isoforms[max_score_idx].features
    psm.site_score = max_score
    psm.site_prob = site_prob
    psm.site_diff_score = diff_score

    # Get the scores of the alternative localization isoforms
    alternatives = []
    for ii in sorted_score_indices[1:]:
        isoform = isoforms[ii]
        sites = [
            site for _, site, mod in isoform.mods if mod == target_mod
            and isinstance(site, int) and psm.seq[site - 1] == target_residue
        ]
        alternatives.append((tuple(sites), scores[ii]))
    psm.alternative_localizations = alternatives


def correct_and_localize(
        psm: PSM,
        target_mod: str,
        target_mod_mass: float,
        target_residue: str,
        model: machinelearning.Classifier,
        features: List[str],
        ptmdb: PTMDB
):
    """
    Corrects isobaric PTMs and localizes `target_mod` in `PSM`.

    Args:
        psm: The peptide spectrum match.
        target_mod: The modification to localize.
        target_mod_mass: The mass change associated with `target_mod`.
        target_residue: The residue targeted by `target_mod`.
        model: The trained machine learning classifier.
        features: The list of features used in validated.
        ptmdb: The unimod PTM database.

    """
    cand_psms = generate_deamidation_candidates(
        psm, ptmdb
    )
    cand_psms.extend(
        generate_alternative_nterm_candidates(
            psm,
            target_mod,
            'Carbamyl',
            ptmdb.get_mass('Carbamyl')
        )
    )
    for cand_psm in cand_psms:
        cand_psm.extract_features()

    if cand_psms:
        # Perform correction by selecting the isoform with the highest
        # score
        feature_array = cand_psms.to_feature_array(features=features)
        isoform_scores = model.predict(feature_array)
        max_isoform_score_idx = np.argmax(isoform_scores)
        validation_scores = model.predict(
            feature_array[max_isoform_score_idx],
            use_cv=True
        )[0]

        # Replace the modifications with the corrected ones
        psm.mods = cand_psms[max_isoform_score_idx].mods
        psm.ml_scores = validation_scores

    if any(ms.mod == target_mod for ms in psm.mods):
        localize(
            psm,
            target_mod,
            target_mod_mass,
            target_residue,
            model,
            features=features
        )
