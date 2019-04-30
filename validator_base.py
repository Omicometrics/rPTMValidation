#! /usr/bin/env python3
"""
This module provides a base class to be inherited for validation and
retrieval pathways.

"""

import collections
import copy
import itertools
import json
import math
import os
import sys

import config
import modifications
import peptides
import plots
import proteolysis
from peptide_spectrum_match import psms2df, UnmodPSM
import readers
import spectra_readers
import utilities

sys.path.append("../pepfrag")
from pepfrag import Peptide


SpecMatch = collections.namedtuple("SpecMatch",
                                   ["seq", "mods", "theor_z", "conf",
                                    "pep_type"])
    
    
class MergedPeptideSequences():
    """
    A class to cache merged peptide sequences to avoid recreating these
    on many iterations.

    """
    def __init__(self):
        """
        Initializes the cache object.

        """
        self._cache = {}
        
    def get(self, seq, mods):
        """
        Retrieves the merged combination of seq and mods. If it does not exist
        in the cache, the merge is performed and cached for future use.
        
        Args:
            seq (str): The peptide sequence.
            mods (list of ModSites): The peptide modification sites.
            
        Returns:
            The peptide string with modifications merged into the sequence.

        """
        key = (seq, tuple(mods))
        
        if key not in self._cache:
            self._cache[key] = peptides.merge_seq_mods(
                seq, mods)
        
        return self._cache[key]
        
        
def site_probability(score, all_scores):
    """
    Computes the probability that the site combination with score is the
    correct combination.

    Args:
        score (float): The current combination LDA score.
        all_scores (list): All of the combination LDA scores.

    Returns:
        The site probability as a float.

        """
    return 1 / sum(math.exp(s) / math.exp(score) for s in all_scores)


class ValidateBase():
    """
    A base class to contain common attributes and methods for validation and
    retrieval pathways for the program.

    """
    def __init__(self, json_config):
        """
        Initialize the object.
        
        Args:
            json_config (json.JSON): The JSON configuration read from a file.

        """
        self.config = config.Config(json_config)

        self.proteolyzer = proteolysis.Proteolyzer(self.config.enzyme)

        # The UniMod PTM DB
        self.unimod = readers.PTMDB(self.config.unimod_ptm_file)
        
        # All ProteinPilot results
        self.pp_res = collections.defaultdict(lambda: collections.defaultdict(list))
        
        self.target_mod = self.config.target_mod
        
        # Get the mass change associated with the target modification
        self.mod_mass = self.unimod.get_mass(self.config.target_mod)
        
    def _identify_benchmarks(self, psms):
        """
        Labels the PSMs which are in the benchmark set of peptides.

        """
        # Parse the benchmark sequences
        with open(self.config.benchmark_file) as fh:
            benchmarks = [l.rstrip() for l in fh]

        for psm in psms:
            psm.benchmark = (peptides.merge_seq_mods(psm.seq, psm.mods)
                             in benchmarks)

    def _find_unmod_analogues(self, mod_psms):
        """
        Finds the unmodified analogues in the ProteinPilot search results.

        Returns:

        """
        unmod_psms = []
        
        merged_seqs = MergedPeptideSequences()
        
        print("Caching PSM sequences...")
        psm_info = []
        for psm in mod_psms:
            mods = [ms for ms in psm.mods if ms.mod != self.target_mod]
            psm_info.append(
                (mods, merged_seqs.get(psm.seq, mods), psm.charge))

        for data_id, data in self.pp_res.items():
            print(f"Processing data set {data_id}...")
            unmods = {}
            for spec_id, matches in data.items():
                pp_peptides = [(merged_seqs.get(match.seq, match.mods),
                                match.theor_z)
                               for match in matches]
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
        
    def localize(self, psms, lda_model, features, spd_threshold,
                 sim_threshold):
        """
        For peptide identifications with multiple possible modification sites,
        localizes the modification site by computing site probabilities.

        """
        for ii, psm in enumerate(psms):
            # Only localize those PSMs which pass the rPTMDetermine score and
            # similarity score thresholds
            if (psm.lda_score < spd_threshold or
                    psm.max_similarity < sim_threshold):
                continue
            
            # Count instances of the free (non-modified) target residues in
            # the peptide sequence
            target_idxs = [jj for jj, res in enumerate(psm.seq)
                           if res in self.config.target_residues]

            # Count the number of instances of the modification
            mod_count = sum(ms.mod == self.target_mod for ms in psm.mods)

            if len(target_idxs) == mod_count:
                # No alternative modification sites exist
                continue

            isoform_scores = {}

            for mod_comb in itertools.combinations(target_idxs, mod_count):
                # Construct a new PSM with the given combination of modified
                # sites
                new_psm = copy.deepcopy(psm)

                # Update the modification list to use the new target sites
                new_psm.mods = [ms for ms in psm.mods
                                if ms.mod != self.target_mod]
                for idx in mod_comb:
                    new_psm.mods.append(
                        modifications.ModSite(
                            self.mod_mass, idx + 1, self.target_mod))

                # Compute the PSM features using the new modification site(s)
                new_psm.extract_features(self.target_mod, self.proteolyzer)

                # Get the target score for the new PSM
                isoform_scores[new_psm] = lda_model.decide_predict(
                    psms2df([new_psm])[features])[0, 0]

            all_scores = list(isoform_scores.values())

            for isoform, score in isoform_scores.items():
                isoform.site_prob = site_probability(score, all_scores)

            psms[ii] = max(isoform_scores.keys(), key=lambda p: p.site_prob)
