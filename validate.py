#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import argparse
from bisect import bisect_left
import collections
import csv
import functools
import itertools
import json
import multiprocessing as mp
import operator
import os
import sys
from typing import List

import numpy as np

from constants import AA_MASSES, FIXED_MASSES, RESIDUES
import decoys
import lda
import modifications
import peptides
import proteolysis
from psm import DecoyID, PSM, psms2df
import readers
import similarity
import utilities

sys.path.append("../pepfrag")
from pepfrag import Peptide


SpecMatch = collections.namedtuple("SpecMatch",
                                   ["seq", "mods", "theor_z", "conf"])


def get_decoys(decoy_db: str, residue: str) -> List:
    """
    Extracts decoy peptides from the database if they contain the specified
    residue.

    Args:
        decoy_db (str): The path to the decoy database.
        residue (str): The residue by which to limit decoy sequences. If None,
                       no residue filter will be applied.

    Returns:
        List of matching decoy peptide sequences.

    """
    with open(decoy_db) as fh:
        rdr = csv.DictReader(fh, delimiter='\t')
        return list({
            r['Sequence'] for r in rdr
            if (residue is None or residue in r['Sequence']) and
            len(r['Sequence']) >= 7 and RESIDUES.issuperset(r['Sequence'])})
            

# TODO: config-based fixed modifications            
def gen_fixed_mods(seq: str) -> List[modifications.ModSite]:
    """
    Generates the fixed modifications for the sequence.

    Args:
        seq (str): The peptide sequence.

    Returns:
        list of fixed ModSites.

    """
    mods = [modifications.ModSite(FIXED_MASSES["tag"], "nterm", "iTRAQ8plex")]
    for ii, res in enumerate(seq):
        # Resolve Lys tag modifications
        if res == 'K':
            mods.append(
                modifications.ModSite(FIXED_MASSES["tag"], ii + 1, None))
        # Resolve Cys carbamidomethylation modifications
        elif res == "C":
            mods.append(modifications.ModSite(FIXED_MASSES["cys_c"],
                                              ii + 1, None))

    return mods


def modify_decoys(seqs, res_idxs, mod_name, mod_mass):
    """
    Applies the target modification to the decoy peptide sequences.

    Args:
        seqs (list): The decoy peptide sequences.
        res_idxs (list of lists): A list of the indices of the residue
                                  targeted by the modification in the peptide.
        mod_name (str): The name of the modification to apply.
        mod_mass (float): The mass associated with the modification.

    Returns:
        tuple: (The indices of the decoy peptides,
                The modifications applied to the decoy peptide,
                The masses of the decoy peptides)

    """
    decoy_idxs, decoy_mods, decoy_seq_masses = [], [], []
    for ii, seq in enumerate(seqs):
        # Calculate the mass of the decoy sequence and construct the
        # modifications
        mods = gen_fixed_mods(seq)
        mass = (FIXED_MASSES["H2O"] +
                sum(AA_MASSES[res].mono for res in seq) +
                sum(ms.mass for ms in mods))

        target_idxs = res_idxs[ii]
        # Generate target modification combinations, up to a maximum of 3
        # instances of the modification
        for jj in range(min(len(target_idxs), 3)):
            for idxs in itertools.combinations(target_idxs, jj + 1):
                decoy_idxs.append(ii)
                decoy_seq_masses.append(mass + mod_mass * len(idxs))
                decoy_mods.append(
                    mods + [modifications.ModSite(mod_mass, kk + 1, mod_name)
                            for kk in idxs])

    return decoy_idxs, decoy_mods, decoy_seq_masses


def match_decoys(peptide_mz, seqs, masses, mods, idxs,
                 var_idxs, slice_idxs, slice_bounds, ptm_masses,
                 ptm_max, ptm_min, tol_factor=0.01):
    """
    """
    candidates = []
    for charge in range(2, 5):
        pep_mass = peptide_mz * charge
        tol = tol_factor * charge

        # start and end are the beginning and ending indices of
        # the slices within which the pep_mass (with a tolerance)
        # falls
        start = slice_idxs[bisect_left(slice_bounds, pep_mass - 1)]
        end = slice_idxs[
            min(bisect_left(slice_bounds, pep_mass + 1), len(slice_idxs) - 1)]

        # Find the decoy sequences which fall within the tolerance
        seq_idxs, = np.asarray(
            (masses[start:end] <= pep_mass + tol) &
            (masses[start:end] >= pep_mass - tol)).nonzero()
        # Shift the indices by the starting index
        seq_idxs += start

        # Add candidate decoy peptides
        candidates.extend(
            [Peptide(seqs[idxs[idx]], charge, mods[idx])
             for idx in seq_idxs])

        # Get new start and end indices accounting for variable
        # PTM masses
        start = slice_idxs[max(
            bisect_left(slice_bounds, pep_mass - ptm_max - 1) - 1, 0)]
        end = slice_idxs[
            min(bisect_left(slice_bounds, pep_mass - ptm_min + 1),
                len(slice_idxs) - 1)]

        # Subset the general decoy lists for the given slice ranges
        r_mods = mods[start:end]
        r_masses = masses[start:end]
        r_seqs = [seqs[idx] for idx in idxs[start:end]]
        r_var_idxs = [var_idxs[idx] for idx in idxs[start:end]]

        for res, _masses in ptm_masses.items():
            for mass in _masses:
                # Find the decoy sequences within the tolerance
                seq_idxs, = np.asarray((r_masses >= pep_mass - tol - mass) &
                                       (r_masses <= pep_mass + tol - mass))\
                                       .nonzero()

                # Find the indices of res in the sequences
                seq_res_idxs = [(ii, jj) for ii in seq_idxs
                                for jj in r_var_idxs[ii]
                                if r_seqs[ii][jj] == res]

                candidates.extend([
                    Peptide(r_seqs[ii],
                            charge, r_mods[ii] +
                            [modifications.ModSite(mass, jj + 1, None)])
                    for ii, jj in seq_res_idxs])

    return candidates


def count_matched_ions(peptide, spectrum):
    """
    Fragments the peptide and counts the number of ions matched against
    the given spectrum.

    Args:
        peptide (pepfrag.Peptide): The peptide to fragment.
        spectrum (Spectrum): The spectrum against which to match ions.

    Returns:
        integer: The number of matching ions between the peptide fragments
                 and the mass spectrum.

    """
    ion_mzs = peptides.get_by_ion_mzs(peptide)
    bisect_idxs = [bisect_left(ion_mzs, mz) for mz in spectrum.mz]
    return sum(
        (idx > 0 and mz - ion_mzs[idx - 1] <= 0.2) or
        (idx < len(ion_mzs) and ion_mzs[idx] - mz <= 0.2)
        for idx, mz in zip(bisect_idxs, list(spectrum.mz)))


def write_results(output_file, psms):
    """
    Writes the PSM results, including features, decoy match features and
    similarity scores, to an output file.

    Args:
        output_file (str): The path to which to write the results.
        psms (list of psm.PSMs): The resulting PSMs.

    """
    feature_names = list(psms[0].features.keys())
    with open(output_file, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter="\t")
        # Write the header row
        writer.writerow(["Rawset", "SpectrumID", "Sequence", "Modifications",
                         "Charge", *[f"Feature_{f}" for f in feature_names],
                         "DecoySequence", "DecoyModifications", "DecoyCharge",
                         *[f"DecoyFeature_{f}" for f in feature_names]])

        # Write the PSM results
        for psm in psms:
            if psm.decoy_id is None:
                continue

            mod_str = ",".join("{:6f}|{}|{}".format(*ms) for ms in psm.mods)
            dmod_str = ",".join("{:6f}|{}|{}".format(*ms)
                                for ms in psm.decoy_id.mods)

            #sim_str = ("none" if psm.similarity_scores is None
            #           else ";".join("{}#{}:{:.6f}".format(sim)
            #                         for sim in psm.similarity_scores))

            # TODO: this way of writing features is horrid,
            # especially since they're now stored in a dict
            writer.writerow([psm.data_id, psm.spec_id, psm.seq, mod_str,
                             psm.charge,
                             *[f"{psm.features[f]:.8f}"
                               for f in feature_names],
                             psm.decoy_id.seq,
                             dmod_str, psm.decoy_id.charge,
                             *[f"{psm.decoy_id.features[f]:.8f}"
                               for f in feature_names]])


def decoy_features(decoy_peptide, spec, target_mod, proteolyzer):
    """
    Calculates the PSM features for the decoy peptide and spectrum
    combination. This function is defined here in order to be picklable
    for multiprocessing.

    """
    return PSM(None, None, decoy_peptide.seq, decoy_peptide.mods,
               decoy_peptide.charge, spectrum=spec).extract_features(
                    target_mod, proteolyzer)
                    
                    
def test_matches_equal(matches, psm, peptide_str) -> bool:
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
        if match.seq != psm.seq and match.theor_z != psm.charge:
            continue

        mods = match.mods

        if "Deamidated" in mods:
            # Remove deamidation from the mods string
            mods = ";".join(mod for mod in mods.split(";")
                            if not mod.startswith("Deamidated"))

        if peptides.merge_seq_mods(match.seq, mods) == peptide_str:
            return True

    return False


class Validate():
    """
    """
    # TODO: implement proper logging
    def __init__(self, json_config):
        """
        """
        self.config = json_config

        self._read_config()

        self.proteolyzer = proteolysis.Proteolyzer(self._enzyme)

        # The UniProt PTM DB
        self.uniprot = readers.read_uniprot_ptms(self._uniprot_ptm_file)
        # The UniMod PTM DB
        self.unimod = readers.PTMDB(self._unimod_ptm_file)

        # Generate the full decoy protein sequence database file
        self.decoy_db_path = decoys.generate_decoy_file(self._target_db_path,
                                                        self.proteolyzer)

        # To be set later
        self.mod_mass = None
        self.psms = None
        self.unmod_psms = None
        self.pp_res = None

        # Used for multiprocessing throughout the class methods
        self.pool = mp.Pool()

    def _read_config(self):
        """
        """
        # TODO: Config class to statically define options

        # The modification for which to validate identifications
        self.target_mod = self.config["modification"]

        # The residues targeted by target_mod
        self.target_residues = self.config["target_residues"]

        # A map of data set IDs to files and confidences.
        self.data_sets = self.config["data_sets"]

        # The path to the UniProt PTM list file.
        self._uniprot_ptm_file = self.config.get("uniprot_ptm_file",
                                                 "ptmlist.txt")
        if not os.path.exists(self._uniprot_ptm_file):
            raise FileNotFoundError("UniProt PTM file not found at "
                                    f"{self._uniprot_ptm_file}")

        # The UniMod PTM DB file path
        self._unimod_ptm_file = self.config.get("unimod_ptm_file",
                                                "unimod.txt")
        if not os.path.exists(self._unimod_ptm_file):
            raise FileNotFoundError("UniMod PTM file not found at "
                                    f"{self._unimod_ptm_file}")

        # The path to the target protein sequence database
        self._target_db_path = self.config["target_database"]
        if not os.path.exists(self._target_db_path):
            raise FileNotFoundError("Target protein sequence database file "
                                    f"not found at {self._target_db_path}")

        # The enzyme rule to be used to digest proteins
        self._enzyme = self.config.get("enzyme", "Trypsin")

        # List of residues bearing fixed modifications.
        self.fixed_residues = self.config["fixed_residues"]

        # The minimum Fisher score for feature selection
        self.fisher_threshold = self.config.get("fisher_score_threshold",
                                                0.05)

    def validate(self):
        """
        """
        # Process the input files to extract the modification identifications
        self.psms, self.pp_res = self._get_identifications()

        # Check whether any modified PSMs are identified
        if not self.psms:
            print("No PSMs found matching the input. Exiting.")
            sys.exit()

        # Get the mass change associated with the target modification
        self.mod_mass = modifications.get_mod_mass(self.psms[0].mods,
                                                   self.target_mod)

        # Read the tandem mass spectra from the raw input files
        # After this call, all PSMs should have their associated mass spectrum
        self.psms = self._process_mass_spectra()

        # Calculate the PSM quality features for each PSM
        for psm in self.psms:
            psm.extract_features(self.target_mod, self.proteolyzer)

        print(f"Total {len(self.psms)} identifications")

        self.psms = list(itertools.chain(
            *[self._generate_decoy_matches(res, self.psms)
              for res in self.target_residues]))
              
        # Write the results to a temporary file
        # TODO: incorporate this properly into the process -
        # if the temporary file exists, use it instead of reprocessing the
        # data
        write_results("test.txt", self.psms)
              
        # Convert the PSMs to a pandas DataFrame, including a "target" column
        # to distinguish target and decoy peptides
        df = psms2df(self.psms)

        # Validate the PSMs using LDA
        results = lda.lda_validate(df, list(self.psms[0].features.keys()),
                                   self.fisher_threshold, cv=10, n_jobs=1)
                                   
        # TODO: only keep the psms whose scores exceed the threshold
        
        # --- Unmodified analogues --- #
        # Get the unmodified peptide analogues
        self.unmod_psms = self._find_unmod_analogues()
        
        # Calculate features for the unmodified peptide analogues
        for psm in self.unmod_psms:
            psm.extract_features(None, self.proteolyzer)
        
        # Add decoy identifications to the unmodified PSMs
        self.unmod_psms = self._generate_decoy_matches(None, self.unmod_psms)
        
        # Validate the unmodified PSMs using LDA
        unmod_df = psms2df(self.unmod_psms)
        unmod_results = lda.lda_validate(
            unmod_df, list(self.unmod_psms[0].features.keys()),
            self.fisher_threshold, cv=10, n_jobs=1)
            
        # TODO: filter unmod PSMs by their validation scores
              
        # --- Similarity Scores --- #
        print("Calculating similarity scores")
        # TODO: use the validated unmodified analogues
        # Calculate the highest similarity score for each target peptide
        self.psms = similarity.calculate_similarity_scores(
            self.psms, self.pp_res, self.target_mod, self.data_sets)

    def _get_identifications(self):
        """
        Retrieves the identification results from the set of input files.

        Returns:
            (list, dict): The PSMs for the target modification and all
                          ProteinPilot results, keyed by input file path.

        """
        # Target modification identifications
        psms = []
        # All ProteinPilot results
        pp_res = collections.defaultdict(lambda: collections.defaultdict(list))

        for set_id, set_info in self.data_sets.items():
            data_dir = set_info['data_dir']
            conf = set_info['confidence']

            summary_files = [os.path.join(data_dir, f)
                             for f in os.listdir(data_dir)
                             if 'PeptideSummary' in f and f.endswith('.txt')]

            for sf in summary_files:
                summaries = readers.read_peptide_summary(
                    sf, condition=lambda r: float(r["Conf"]) >= conf)
                for summary in summaries:
                    mods = modifications._preparse_mod_string(summary.mods)

                    try:
                        parsed_mods = modifications.parse_mods(
                            mods, self.unimod)
                    except modifications.UnknownModificationException:
                        continue

                    if any(f"{self.target_mod}({tr})" in summary.mods
                           for tr in self.target_residues):
                        psms.append(
                            PSM(set_id, summary.spec, summary.seq,
                                parsed_mods, summary.theor_z))

                    pp_res[set_id][summary.spec].append(
                        SpecMatch(summary.seq, mods, summary.theor_z,
                                  summary.conf))

        return psms, pp_res

    def _process_mass_spectra(self):
        """
        Processes the input mass spectra to match to their peptides.

        Returns:
            The PSM objects, now with their associated mass spectra.

        """
        for set_id, config in self.data_sets.items():
            spec_file = os.path.join(config['data_dir'],
                                     config['spectra_file'])

            if not os.path.isfile(spec_file):
                raise FileNotFoundError(f"Spectra file {spec_file} not found")

            spectra = readers.read_spectra_file(spec_file)

            for psm in self.psms:
                if psm.data_id == set_id and psm.spec_id in spectra:
                    psm.spectrum = spectra[psm.spec_id]
                    psm.spectrum = psm.spectrum.centroid().remove_itraq()

        return self.psms
        
    def _find_unmod_analogues(self):
        """
        Finds the unmodified analogues in the ProteinPilot search results.
        
        Returns:
            
        """
        # TODO: store the unmod psms mapped to their corresponding mod psms
        # TODO: store spectra in memory?
        unmod_psms = []
        for data_id, data in self.pp_res.items():
            unmods = collections.defaultdict(list)
            for spec_id, matches in data.items():
                for psm in self.psms:
                    mods = [ms for ms in psm.mods if ms.mod != self.target_mod]
                    peptide_str = peptides.merge_seq_mods(psm.seq, mods)
                    if test_matches_equal(matches, psm, peptide_str):
                        unmods[spec_id].append((psm, mods))
                        
            if not unmods:
                continue

            spec_file = os.path.join(self.data_sets[data_id]["data_dir"],
                                     self.data_sets[data_id]["spectra_file"])

            print(f"Reading {spec_file}")
            spectra = readers.read_spectra_file(spec_file)
            
            print(f"Processing {len(unmods)} spectra")

            for spec_id, _psms in unmods.items():
                print(f"Processing {spec_id} with {_psms}")
                spec = spectra[spec_id].centroid().remove_itraq()

                for psm, mods in _psms:
                    unmod_psms.append(PSM(data_id, spec_id, psm.seq,
                                          mods, psm.charge, spectrum=spec))
                                          
        return unmod_psms

    def _generate_decoy_matches(self, target_res, psms):
        """
        
        Args:
            target_res (str): The target (fixed) residue. If None, all
                              residues not contained in self.fixed_residues
                              are subject to variable modifications.
            psms (list of PSMs):

        """
        # The residues bearing "fixed" modifications
        fixed_aas = list(self.fixed_residues)  # list used to ensure copy
        if target_res is not None:
            fixed_aas.append(target_res)

        # Generate the decoy sequences, including the target_mod if
        # target_residue is provided
        d_seqs, d_var_idxs, d_idxs, d_mods, d_masses = \
            self._generate_residue_decoys(target_res, fixed_aas)

        # Convert to numpy array for use with np.asarray
        d_masses = np.array(d_masses)

        # Split the decoy mass range into slices of 500 peptides to optimize
        # the search
        slice_idxs, slice_bounds = \
            utilities.slice_list(d_masses, nslices=int(len(d_masses) / 500))

        msg = f"Generated {len(d_seqs)} random sequences"
        if target_res is not None:
            msg += f" for target residue {target_res}"
        print(msg)

        # Dictionary of AA residue to list of possible modification masses
        var_ptm_masses = {res: list({m[1] for m in mods if m[1] is not None
                                     and abs(m[1]) <= 100})
                          for res, mods in self.uniprot.items()
                          if res not in fixed_aas}
        var_ptm_max = max(max(masses) for masses in var_ptm_masses.values())
        var_ptm_min = min(min(masses) for masses in var_ptm_masses.values())

        def _match_decoys(peptide_mz, tol_factor):
            return match_decoys(peptide_mz, d_seqs, d_masses, d_mods, d_idxs,
                                d_var_idxs, slice_idxs, slice_bounds,
                                var_ptm_masses, var_ptm_max, var_ptm_min,
                                tol_factor=tol_factor)
                                
        pep_strs = [peptides.merge_seq_mods(psm.seq, psm.mods)
                    for psm in psms]

        # Deduplicate peptide list
        pep_strs_set = set(pep_strs)

        for ii, peptide in enumerate(pep_strs_set):
            print(f"Processing peptide {ii + 1} of {len(pep_strs_set)} "
                  f"- {peptide}")
            # Find the indices of the peptide in peptides
            pep_idxs = [idx for idx, pep in enumerate(pep_strs)
                        if pep == peptide]

            mods = psms[pep_idxs[0]].mods

            # Find all of the charge states for the peptide
            charge_states = {psms[idx].charge for idx in pep_idxs}

            for charge in charge_states:
                # Find the indices of the matching peptides with the given
                # charge state
                charge_pep_idxs =\
                    [jj for jj in pep_idxs if psms[jj].charge == charge]

                # Calculate the mass/charge ratio of the peptide, using the PSM
                # of the first instance of this peptide with this charge
                pep_mz = peptides.calculate_mz(
                    psms[charge_pep_idxs[0]].seq, mods, charge)

                # Get the spectra associated with the peptide
                spectra = [(psms[idx].spectrum,
                            psms[idx].spectrum.max_intensity(),
                            idx) for idx in charge_pep_idxs
                           if psms[idx].spectrum]

                if not spectra:
                    continue

                # Extract the spectrum with the highest base peak intensity
                max_spec = max(spectra, key=operator.itemgetter(1))[0]

                # Generate decoy candidate peptides by searching the mass
                # slices
                d_candidates = _match_decoys(pep_mz, tol_factor=0.01)

                if len(d_candidates) < 1000:
                    # Search again using a larger mass tolerance
                    d_candidates = _match_decoys(pep_mz, tol_factor=0.1)

                if len(d_candidates) == 0:
                    continue

                # Find the number of matched ions in the spectrum per decoy
                # peptide candidate
                _count_matched_ions = functools.partial(count_matched_ions,
                                                        spectrum=max_spec)
                cand_num_ions = self.pool.map(_count_matched_ions,
                                              d_candidates)

                # Order the decoy matches by the number of ions matched
                sorted_idxs = sorted(range(len(cand_num_ions)),
                                     key=lambda k: cand_num_ions[k],
                                     reverse=True)

                # Keep only the top 1000 decoy candidates in terms of the
                # the number of ions matched
                d_candidates = [d_candidates[jj] for jj in sorted_idxs[:1000]]

                # For each spectrum, find the top matching decoy peptide
                # and calculate the features for the match
                for jj, (spec, _, idx) in enumerate(spectra):
                    _decoy_features = functools.partial(
                        decoy_features, spec=spec,
                        target_mod=self.target_mod if target_res is not None
                                   else None,
                        proteolyzer=self.proteolyzer)
                    dpsm_vars = self.pool.map(_decoy_features, d_candidates)

                    # Find the decoy candidate with the highest MatchScore
                    max_match = max(dpsm_vars, key=lambda k: k["MatchScore"])

                    # If the decoy ID is better than the one already assigned
                    # to the PSM, then replace it
                    if (psms[idx].decoy_id is None or
                            psms[idx].decoy_id.features["MatchScore"] <
                            max_match["MatchScore"]):
                        d_peptide = d_candidates[dpsm_vars.index(max_match)]
                        psms[idx].decoy_id = \
                            DecoyID(d_peptide.seq, d_peptide.charge,
                                    d_peptide.mods, max_match)

        return psms

    def _generate_residue_decoys(self, target_res, fixed_aas):
        """
        Generate the base decoy peptides with fixed modifications applied,
        including the target modification at target_res if specified.
        
        Args:
            target_res (str): The target (fixed) residue. If None, all
                              residues not contained in self.fixed_residues
                              are subject to variable modifications.
            fixed_aas (list): The amino acid residues which should bear fixed
                              modifications.
        
        Returns:
            
        """
        # Generate list of decoy peptides containing the residue of interest
        seqs = get_decoys(self.decoy_db_path, target_res)

        # Extract the indices of the residues in the decoy peptides which are
        # not modified by a fixed modification
        var_idxs = [
            [idx for idx, res in enumerate(seq) if res not in fixed_aas]
            for seq in seqs]

        if target_res is not None:
            # Find the sites of the target residue in the decoy peptides
            res_idxs = [[idx for idx, res in enumerate(seq) if res == target_res]
                        for seq in seqs]

            # Apply the target modification to the decoy peptides
            idxs, mods, masses = modify_decoys(seqs, res_idxs, self.target_mod,
                                               self.mod_mass)
        else:
            # For unmodified analogues, apply the fixed modifications and
            # calculate the peptide masses
            idxs = list(range(len(seqs)))
            mods, masses = [], []
            for seq in seqs:
                _mods = gen_fixed_mods(seq)
                mods.append(_mods)
                masses.append(FIXED_MASSES["H2O"] +
                              sum(AA_MASSES[res].mono for res in seq) +
                              sum(ms.mass for ms in _mods))

        # Sort the sequence masses, indices and mods according to the
        # sequence mass
        masses, idxs, mods = utilities.sort_lists(0, masses, idxs, mods)

        return seqs, var_idxs, idxs, mods, masses


def parse_args():
    """
    Parses the command line arguments to the script.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help=("The path to the JSON configuration file. "
              "See example_input.json for an example"))
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as fh:
        config = json.load(fh)

    Validate(config).validate()


if __name__ == '__main__':
    main()
