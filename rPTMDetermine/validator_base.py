#! /usr/bin/env python3
"""
This module provides a base class to be inherited for validation and
retrieval pathways.

"""

import collections
import copy
import functools
import itertools
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

from pepfrag import ModSite, Peptide

from . import lda
from . import peptides
from . import proteolysis
from .peptide_spectrum_match import PSM, UnmodPSM
from .psm_container import PSMContainer
from . import readers
from .readers.base_reader import Reader
from . import spectra_readers
from . import utilities


SpecMatch = collections.namedtuple("SpecMatch",
                                   ["seq", "mods", "theor_z", "conf",
                                    "pep_type"])


@functools.lru_cache(maxsize=1024)
def merge_peptide_sequence(seq: str, mods: Tuple[ModSite, ...]) -> str:
    """
    Merges the modifications into the peptide sequence.

    Args:
        seq (str): The peptide sequence.
        mods (tuple): The peptide ModSites.

    Returns:
        The merged sequence as a string.

    """
    return peptides.merge_seq_mods(seq, mods)


def site_probability(score: float, all_scores: Iterable[float]) -> float:
    """
    Computes the probability that the site combination with score is the
    correct combination.

    Args:
        score (float): The current combination LDA score.
        all_scores (list): All of the combination LDA scores.

    Returns:
        The site probability as a float.

    """
    return 1. / sum(math.exp(s) / math.exp(score) for s in all_scores)


class ValidateBase():
    """
    A base class to contain common attributes and methods for validation and
    retrieval pathways for the program.

    """
    def __init__(self, config):
        """
        Initialize the object.

        Args:
            config (subclass of BaseConfig)

        """
        # Config does not have a type hint to allow duck typing of
        # ValidatorConfig and RetrieverConfig
        self.config = config

        self.proteolyzer = proteolysis.Proteolyzer(self.config.enzyme)

        # The UniMod PTM DB
        print("Reading UniMod PTM DB")
        self.unimod = readers.PTMDB(self.config.unimod_ptm_file)

        # The database search reader
        self.reader: Reader = readers.get_reader(
            self.config.search_engine, self.unimod)

        # All database search results
        self.db_res: Dict[str, Dict[str, List[SpecMatch]]] =\
            collections.defaultdict(lambda: collections.defaultdict(list))

        self.target_mod = self.config.target_mod

        # Get the mass change associated with the target modification
        self.mod_mass = self.unimod.get_mass(self.config.target_mod)

        path_str = (f"{self.target_mod.replace('->', '2')}_"
                    f"{''.join(self.config.target_residues)}")

        output_dir = self.config.output_dir
        if output_dir is None:
            output_dir = path_str

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.file_prefix = f"{output_dir}/{path_str}_"

    def identify_benchmarks(self, psms: Sequence[PSM]):
        """
        Labels the PSMs which are in the benchmark set of peptides.

        """
        if self.config.benchmark_file is not None:
            # Parse the benchmark sequences
            with open(self.config.benchmark_file) as fh:
                benchmarks = [l.rstrip() for l in fh]

            for psm in psms:
                psm.benchmark = (peptides.merge_seq_mods(psm.seq, psm.mods)
                                 in benchmarks)

    def _find_unmod_analogues(self, mod_psms: Sequence[PSM]):
        """
        Finds the unmodified analogues in the database search results.

        Returns:

        """
        unmod_psms = []

        print("Caching PSM sequences...")
        psm_info = []
        for psm in mod_psms:
            mods = []
            for mod in psm.mods:
                try:
                    site = int(mod.site)
                except ValueError:
                    mods.append(mod)
                    continue

                if (mod.mod != self.target_mod and
                        psm.seq[site - 1] not in self.config.target_residues):
                    mods.append(mod)

            psm_info.append(
                (mods, merge_peptide_sequence(psm.seq, tuple(mods)),
                 psm.charge))

        for data_id, data in self.db_res.items():
            print(f"Processing data set {data_id}...")
            unmods = {}
            for spec_id, matches in data.items():
                pp_peptides = [(
                    merge_peptide_sequence(match.seq, tuple(match.mods)),
                    match.theor_z) for match in matches]
                res = [(mod_psms[idx], mods)
                       for idx, (mods, pep_str, charge) in enumerate(psm_info)
                       if (pep_str, charge) in pp_peptides]
                if res:
                    unmods[spec_id] = res

            if not unmods:
                continue

            spec_file = os.path.join(
                self.config.data_sets[data_id]["data_dir"],
                self.config.data_sets[data_id]["spectra_file"])

            print(f"Reading {spec_file}...")
            spectra = spectra_readers.read_spectra_file(spec_file)

            print(f"Processing {len(unmods)} spectra...")

            for spec_id, _psms in unmods.items():
                spec = spectra[spec_id].centroid().remove_itraq()

                for psm, mods in _psms:
                    unmod_psms.append(
                        UnmodPSM(psm.uid, data_id, spec_id,
                                 Peptide(psm.seq, psm.charge, mods),
                                 spectrum=spec))

        return utilities.deduplicate(unmod_psms)

    def _filter_mods(self, mods: Iterable[ModSite], seq: str)\
            -> List[ModSite]:
        """
        Filters the modification list to remove those instances of the
        target_mod at one of the target_residues.

        Args:
            mods (list of ModSites)
            seq (str): The peptide sequence.

        Returns:
            Filtered list of ModSites.

        """
        new_mods = []
        for mod_site in mods:
            try:
                site = int(mod_site.site)
            except ValueError:
                new_mods.append(mod_site)
                continue

            if (mod_site.mod != self.target_mod and
                    seq[site - 1] not in self.config.target_residues):
                new_mods.append(mod_site)

        return new_mods

    def _localize(self, psms: List[PSM], lda_model: lda.CustomPipeline,
                  features: Iterable[str], spd_prob_threshold: float,
                  sim_threshold: float):
        """
        For peptide identifications with multiple possible modification sites,
        localizes the modification site by computing site probabilities.

        """
        for ii, psm in enumerate(psms):
            # Only localize those PSMs which pass the rPTMDetermine score and
            # similarity score thresholds
            if (psm.lda_prob is None or psm.lda_prob < spd_prob_threshold or
                    psm.max_similarity < sim_threshold):
                continue

            # Count instances of the free (non-modified) target residues in
            # the peptide sequence
            target_idxs = [jj for jj, res in enumerate(psm.seq)
                           if res in self.config.target_residues +
                           self.config.alternative_localization_residues]

            # Count the number of instances of the modification
            mod_count =\
                sum(ms.mod == self.target_mod and
                    isinstance(ms.site, int) and
                    (ms.site - 1) in target_idxs
                    for ms in psm.mods)

            if len(target_idxs) == mod_count:
                # No alternative modification sites exist
                continue

            isoform_scores = {}

            for mod_comb in itertools.combinations(target_idxs, mod_count):
                # Construct a new PSM with the given combination of modified
                # sites
                new_psm = copy.deepcopy(psm)

                # Update the modification list to use the new target sites
                new_psm.mods = self._filter_mods(new_psm.mods, new_psm.seq)
                for idx in mod_comb:
                    new_psm.mods.append(
                        ModSite(
                            self.mod_mass, idx + 1, self.target_mod))

                # Compute the PSM features using the new modification site(s)
                new_psm.extract_features(self.target_mod, self.proteolyzer)

                # Get the target score for the new PSM
                isoform_scores[new_psm] = lda_model.decide_predict(
                    PSMContainer([new_psm]).to_df()[features])[0, 0]

            all_scores = list(isoform_scores.values())

            for isoform, score in isoform_scores.items():
                isoform.site_prob = site_probability(score, all_scores)

            psms[ii] = max(isoform_scores.keys(), key=lambda p: p.site_prob)

    def filter_localizations(self, psms: Sequence[PSM]) -> PSMContainer:
        """
        Filters the PSMs to retain only those which have the target_mod
        localized to one of the target residues.

        Args:
            psms (list of PSMs): The PSMs to be filtered.

        Returns:
            Filtered list of PSMs.

        """
        new_psms = []
        for psm in psms:
            seq = psm.seq
            for mod_site in psm.mods:
                if mod_site.site == "nterm" or mod_site.site == "cterm":
                    continue
                if (mod_site.mod == self.target_mod and
                        seq[int(mod_site.site) - 1] in
                        self.config.target_residues):
                    new_psms.append(psm)
                    break
        return PSMContainer(new_psms)
