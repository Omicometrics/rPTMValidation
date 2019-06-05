#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
from bisect import bisect_left
import collections
import copy
import csv
import functools
import itertools
import operator
import os
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tqdm

from pepfrag import AA_MASSES, FIXED_MASSES, ModSite, Peptide

from .constants import RESIDUES
from . import generate_decoys
from . import lda
from . import mass_spectrum
from . import peptides
from .peptide_spectrum_match import DecoyID, PSM
from . import proteolysis
from .psm_container import PSMContainer
from . import readers
from . import similarity
from . import utilities
from . import validator_base
from .validator_config import ValidatorConfig


DecoyPeptides = collections.namedtuple("DecoyPeptides",
                                       ["seqs", "var_idxs", "idxs", "mods",
                                        "masses"])


VarPTMs = collections.namedtuple("VarPTMs", ["masses", "max_mass",
                                             "min_mass"])


def get_decoys(decoy_db: str, residue: Optional[str]) -> List[str]:
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
    with open(decoy_db) as handle:
        rdr = csv.DictReader(handle, delimiter='\t')
        return list({
            r['Sequence'] for r in rdr
            if (residue is None or residue in r['Sequence']) and
            len(r['Sequence']) >= 7 and RESIDUES.issuperset(r['Sequence'])})


def match_decoys(peptide_mz: float, decoys: DecoyPeptides,
                 slices: utilities.Slices, var_ptms: VarPTMs,
                 tol_factor: float = 0.01) -> List[Peptide]:
    """
    Finds the decoy peptide candidates for the given peptide mass charge
    ratio.

    Args:
    """
    candidates = []
    for charge in range(2, 5):
        pep_mass = peptide_mz * charge
        tol = tol_factor * charge

        # start and end are the beginning and ending indices of
        # the slices within which the pep_mass (with a tolerance)
        # falls
        start = slices.idxs[bisect_left(slices.bounds, pep_mass - 1)]
        end = slices.idxs[
            min(bisect_left(slices.bounds, pep_mass + 1),
                len(slices.idxs) - 1)]

        # Find the decoy sequences which fall within the tolerance
        seq_idxs, = np.asarray(
            (decoys.masses[start:end] <= pep_mass + tol) &
            (decoys.masses[start:end] >= pep_mass - tol)).nonzero()
        # Shift the indices by the starting index
        seq_idxs += start

        # Add candidate decoy peptides
        candidates.extend(
            [Peptide(decoys.seqs[decoys.idxs[idx]], charge, decoys.mods[idx])
             for idx in seq_idxs])

        # Get new start and end indices accounting for variable
        # PTM masses
        start = slices.idxs[max(
            bisect_left(slices.bounds, pep_mass - var_ptms.max_mass - 1) - 1,
            0)]
        end = slices.idxs[
            min(bisect_left(slices.bounds, pep_mass - var_ptms.min_mass + 1),
                len(slices.idxs) - 1)]

        # Subset the general decoy lists for the given slice ranges
        r_mods = decoys.mods[start:end]
        r_masses = decoys.masses[start:end]
        r_seqs = [decoys.seqs[idx] for idx in decoys.idxs[start:end]]
        r_var_idxs = [decoys.var_idxs[idx] for idx in decoys.idxs[start:end]]

        for res, _masses in var_ptms.masses.items():
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
                            [ModSite(mass, jj + 1, None)])
                    for ii, jj in seq_res_idxs])

    return candidates


def count_matched_ions(peptide: Peptide,
                       spectrum: mass_spectrum.Spectrum) -> int:
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


def write_results(output_file: str, psms: Sequence[PSM],
                  include_features: bool = False):
    """
    Writes the PSM results, including features, decoy match features and
    similarity scores, to an output file.

    Args:
        output_file (str): The path to which to write the results.
        psms (list of peptide_spectrum_match.PSMs): The resulting PSMs.

    """
    feature_names = list(psms[0].features.keys())
    with open(output_file, 'w', newline='') as handle:
        writer = csv.writer(handle, delimiter="\t")
        # Write the header row
        header = ["Rawset", "SpectrumID", "Sequence", "Modifications",
                  "Charge", "DecoySequence", "DecoyModifications",
                  "DecoyCharge", "rPTMDetermineScore", "rPTMDetermineProb",
                  "SimilarityScore", "SiteProbability"]
        if include_features:
            header.extend([f"Feature_{f}" for f in feature_names])
            header.extend([f"DecoyFeature_{f}" for f in feature_names])

        writer.writerow(header)

        # Write the PSM results
        for psm in psms:
            if psm.decoy_id is None:
                continue

            mod_str = ",".join("{:6f}|{}|{}".format(*ms) for ms in psm.mods)
            dmod_str = ",".join("{:6f}|{}|{}".format(*ms)
                                for ms in psm.decoy_id.mods)

            row = [psm.data_id, psm.spec_id, psm.seq, mod_str, psm.charge,
                   psm.decoy_id.seq, dmod_str, psm.decoy_id.charge,
                   psm.lda_score, psm.lda_prob, psm.max_similarity,
                   psm.site_prob if psm.site_prob is not None else ""]

            if include_features:
                row.extend([f"{psm.features[f]:.8f}" for f in feature_names])
                row.extend([f"{psm.decoy_id.features[f]:.8f}"
                            for f in feature_names])

            writer.writerow(row)


def decoy_features(decoy_peptide: Peptide, spec: mass_spectrum.Spectrum,
                   target_mod: str,
                   proteolyzer: proteolysis.Proteolyzer) -> Dict[str, float]:
    """
    Calculates the PSM features for the decoy peptide and spectrum
    combination. This function is defined here in order to be picklable
    for multiprocessing.

    """
    return PSM(None, None, decoy_peptide, spectrum=copy.deepcopy(spec))\
        .extract_features(target_mod, proteolyzer)


class Validator(validator_base.ValidateBase):
    """
    The main rPTMDetermine class. The validate method of this class
    encompasses the main functionality of the procedure.

    """
    # TODO: implement proper logging
    def __init__(self, json_config):
        """
        Initialize the Validate object.

        Args:
            json_config (json.JSON): The JSON configuration read from a file.

        """
        super().__init__(ValidatorConfig(json_config))

        # The UniProt PTM DB
        self.uniprot = readers.read_uniprot_ptms(self.config.uniprot_ptm_file)

        # Generate the full decoy protein sequence database file
        self.decoy_db_path = generate_decoys.generate_decoy_file(
            self.config.target_db_path, self.proteolyzer)

        # Cache these config options since they are used regularly
        self.target_residues = self.config.target_residues
        self.fixed_residues = self.config.fixed_residues

        # To be set later
        self.psms: List[PSM] = []
        self.unmod_psms = None

        # The LDA validation model for scoring
        self.model = None
        self.mod_features = None

    def validate(self):
        """
        Validates the identifications in the input data files.

        """
        # Process the input files to extract the modification identifications
        print("Reading database search identifications...")
        self.psms = self._get_identifications()

        # Check whether any modified PSMs are identified
        if not self.psms:
            print("No PSMs found matching the input. Exiting.")
            sys.exit()

        # Read the tandem mass spectra from the raw input files
        # After this call, all PSMs will have their associated mass spectrum
        print("Reading mass spectra from files...")
        self.psms = self._process_mass_spectra()

        # Calculate the PSM quality features for each PSM
        print("Calculating PSM features...")
        for psm in tqdm.tqdm(self.psms):
            psm.extract_features(self.target_mod, self.proteolyzer)

        print(f"Total {len(self.psms)} identifications")

        print("Generating decoy PSMs...")
        self.psms = list(itertools.chain(
            *[self._generate_decoy_matches(res, self.psms)
              for res in self.target_residues]))

        # Convert the PSMs to a pandas DataFrame, including a "target" column
        # to distinguish target and decoy peptides
        mod_df = PSMContainer(self.psms).to_df()

        # Validate the PSMs using LDA
        print("Validating PSMs...")
        self.mod_features = [f for f in list(self.psms[0].features.keys())
                             if f not in self.config.exclude_features]

        # Train full LDA model
        self.model, score_stats, full_lda_threshold =\
            lda.lda_model(mod_df, self.mod_features)

        results = lda.lda_validate(mod_df, self.mod_features,
                                   full_lda_threshold)

        # Merge the LDA results to the PSM objects
        self.psms = lda.merge_lda_results(self.psms, results)

        # Apply a deamidation removal: test whether the non-deamidated
        # peptide has a higher score for the identification
        if self.config.correct_deamidation:
            print("Applying deamidation correction...")
            self.psms = lda.apply_deamidation_correction(
                self.model, score_stats, self.psms, self.mod_features,
                self.target_mod, self.proteolyzer)

        # Write the model input to a file for re-use in retrieval
        mod_df.to_csv(self.file_prefix + "model.csv")

        # Identify the PSMs whose peptides are benchmarks
        if self.config.benchmark_file is not None:
            self.identify_benchmarks(self.psms)

        # --- Unmodified analogues --- #
        # Get the unmodified peptide analogues
        print("Finding unmodified analogues...")
        self.unmod_psms = self._find_unmod_analogues(self.psms)

        # Calculate features for the unmodified peptide analogues
        print("Calculating unmodified PSM features...")
        for psm in tqdm.tqdm(self.unmod_psms):
            psm.extract_features(None, self.proteolyzer)

        # Add decoy identifications to the unmodified PSMs
        print("Generating decoy matches for unmodified analogues...")
        self.unmod_psms = self._generate_decoy_matches(None, self.unmod_psms)

        # Validate the unmodified PSMs using LDA
        unmod_df = PSMContainer(self.unmod_psms).to_df()

        print("Validating unmodified analogues...")
        unmod_features = [f for f in list(self.unmod_psms[0].features.keys())
                          if f not in self.config.exclude_features]

        unmod_model, unmod_score_stats, unmod_lda_threshold =\
            lda.lda_model(unmod_df, unmod_features)

        unmod_results =\
            lda.lda_validate(unmod_df, unmod_features, unmod_lda_threshold)

        self.unmod_psms =\
            lda.merge_lda_results(self.unmod_psms, unmod_results)

        if self.config.correct_deamidation:
            print("Applying deamidation correction for unmodified analogues")
            self.unmod_psms = lda.apply_deamidation_correction(
                unmod_model, unmod_score_stats, self.unmod_psms,
                unmod_features, None, self.proteolyzer)

        unmod_df.to_csv(self.file_prefix + "unmod_model.csv")

        with open(self.file_prefix + "unmod_psms", "wb") as fh:
            pickle.dump(self.unmod_psms, fh)

        # Filter the unmodified analogues according to their probabilities
        self.unmod_psms = [p for p in self.unmod_psms if p.lda_prob >= 0.99]

        # --- Similarity Scores --- #
        print("Calculating similarity scores...")
        # Calculate the highest similarity score for each target peptide
        self.psms = similarity.calculate_similarity_scores(self.psms,
                                                           self.unmod_psms)

    def localize(self):
        """
        For peptide identifications with multiple possible modification sites,
        localizes the modification site by computing site probabilities.

        """
        if self.config.sim_threshold_from_benchmarks:
            sim_threshold = min(psm.max_similarity for psm in self.psms
                                if psm.benchmark)
        else:
            sim_threshold = self.config.sim_threshold
        print("Localizing modification sites...")
        super()._localize(self.psms, self.model, self.mod_features,
                          0.99, sim_threshold)
        self.psms = self.filter_localizations(self.psms)

    def _get_identifications(self) -> List[PSM]:
        """
        Retrieves the identification results from the set of input files.

        Returns:
            list: The PSMs for the target modification.

        """
        # Target modification identifications
        psms = []
        for set_id, set_info in tqdm.tqdm(self.config.data_sets.items()):
            res_path = os.path.join(set_info["data_dir"], set_info["results"])

            # Apply database search FDR control to the results
            identifications: Sequence[readers.SearchResult] = \
                self.reader.read(res_path,
                                 predicate=self._fdr_filter(set_info))

            for ident in identifications:
                if (ident.pep_type == readers.PeptideType.decoy or
                        ident.rank != 1 or "X" in ident.seq):
                    # Filter to rank 1, target identifications for validation
                    # and ignore placeholder amino acid residue identifications
                    continue

                dataset = (ident.dataset if ident.dataset is not None
                           else set_id)

                if any(ms.mod == self.target_mod and isinstance(ms.site, int)
                       and ident.seq[ms.site - 1] == res for ms in ident.mods
                       for res in self.target_residues):
                    psms.append(
                        PSM(dataset, ident.spectrum,
                            Peptide(ident.seq, ident.charge, ident.mods)))

                self.db_res[dataset][ident.spectrum].append(ident)

        return utilities.deduplicate(psms)

    def _process_mass_spectra(self) -> List[PSM]:
        """
        Processes the input mass spectra to match to their peptides.

        Returns:
            The PSM objects, now with their associated mass spectra.

        """
        all_spectra = self.read_mass_spectra()
        for set_id, spectra in all_spectra.items():
            for psm in self.psms:
                if psm.data_id == set_id and psm.spec_id in spectra:
                    psm.spectrum = spectra[psm.spec_id]

        return self.psms

    def _generate_decoy_matches(self, target_res: Optional[str],
                                psms: List[PSM]) -> List[PSM]:
        """

        Args:
            target_res (str): The target (fixed) residue. If None, all
                              residues not contained in self.fixed_residues
                              are subject to variable modifications.
            psms (list of PSMs):

        """
        # The residues bearing "fixed" modifications
        fixed_aas = list(self.fixed_residues.keys())
        if target_res is not None:
            fixed_aas.append(target_res)
        # Remove termini
        for pos in ["nterm", "cterm"]:
            if pos in fixed_aas:
                fixed_aas.remove(pos)

        # Generate the decoy sequences, including the target_mod if
        # target_residue is provided
        decoys = self._generate_residue_decoys(target_res, fixed_aas)

        # Split the decoy mass range into slices of 500 peptides to optimize
        # the search
        slices = \
            utilities.slice_list(decoys.masses,
                                 nslices=int(len(decoys.masses) / 500))

        msg = f"Generated {len(decoys.seqs)} random sequences"
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

        var_ptms = VarPTMs(var_ptm_masses, var_ptm_max, var_ptm_min)

        def _match_decoys(peptide_mz, tol_factor):
            return match_decoys(peptide_mz, decoys, slices, var_ptms,
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

                if not d_candidates:
                    continue

                # Find the number of matched ions in the spectrum per decoy
                # peptide candidate
                _count_matched_ions = functools.partial(count_matched_ions,
                                                        spectrum=max_spec)
                cand_num_ions =\
                    self.pool.map(_count_matched_ions, d_candidates)

                # Order the decoy matches by the number of ions matched
                sorted_idxs: List[int] = sorted(
                    range(len(cand_num_ions)),
                    key=lambda k, cand_ions=cand_num_ions: cand_ions[k],
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
                    psm = psms[idx]
                    if (psm.decoy_id is None or
                            psm.decoy_id.features["MatchScore"] <
                            max_match["MatchScore"]):
                        d_peptide = d_candidates[dpsm_vars.index(max_match)]
                        psm.decoy_id = \
                            DecoyID(d_peptide.seq, d_peptide.charge,
                                    d_peptide.mods, max_match)

        return psms

    def _generate_residue_decoys(self, target_res: Optional[str],
                                 fixed_aas: List[str]) -> DecoyPeptides:
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
            DecoyPeptides

        """
        # Generate list of decoy peptides containing the residue of interest
        seqs = get_decoys(self.decoy_db_path, target_res)

        # Extract the indices of the residues in the decoy peptides which are
        # not modified by a fixed modification
        var_idxs = [
            [idx for idx, res in enumerate(seq) if res not in fixed_aas]
            for seq in seqs]

        idxs: List[int] = []
        mods: List[List[ModSite]] = []
        masses: List[float] = []
        if target_res is not None:
            # Find the sites of the target residue in the decoy peptides
            res_idxs = [[idx for idx, res in enumerate(seq)
                         if res == target_res]
                        for seq in seqs]

            # Apply the target modification to the decoy peptides
            idxs, mods, masses = self.modify_decoys(seqs, res_idxs)
        else:
            # For unmodified analogues, apply the fixed modifications and
            # calculate the peptide masses
            idxs = list(range(len(seqs)))
            for seq in seqs:
                _mods = self.gen_fixed_mods(seq)
                mods.append(_mods)
                masses.append(FIXED_MASSES["H2O"] +
                              sum(AA_MASSES[res].mono for res in seq) +
                              sum(ms.mass for ms in _mods))

        # Sort the sequence masses, indices and mods according to the
        # sequence mass
        masses, idxs, mods = utilities.sort_lists(0, masses, idxs, mods)

        return DecoyPeptides(seqs, var_idxs, idxs, mods, np.array(masses))

    def modify_decoys(self, seqs: List[str], res_idxs: List[List[int]])\
            -> Tuple[List[int], List[List[ModSite]], List[float]]:
        """
        Applies the target modification to the decoy peptide sequences.

        Args:
            seqs (list): The decoy peptide sequences.
            res_idxs (list of lists): A list of the indices of the residue
                                      targeted by the modification in the
                                      peptide.

        Returns:
            tuple: (The indices of the decoy peptides,
                    The modifications applied to the decoy peptide,
                    The masses of the decoy peptides)

        """
        decoy_idxs: List[int] = []
        decoy_mods: List[List[ModSite]] = []
        decoy_seq_masses: List[float] = []
        for ii, seq in enumerate(seqs):
            # Calculate the mass of the decoy sequence and construct the
            # modifications
            mods = self.gen_fixed_mods(seq)
            mass = (FIXED_MASSES["H2O"] +
                    sum(AA_MASSES[res].mono for res in seq) +
                    sum(ms.mass for ms in mods))

            target_idxs = res_idxs[ii]
            # Generate target modification combinations, up to a maximum of 3
            # instances of the modification
            for jj in range(min(len(target_idxs), 3)):
                for idxs in itertools.combinations(target_idxs, jj + 1):
                    decoy_idxs.append(ii)
                    decoy_seq_masses.append(mass + self.mod_mass * len(idxs))
                    decoy_mods.append(
                        mods + [ModSite(self.mod_mass, kk + 1,
                                        self.target_mod)
                                for kk in idxs])

        return decoy_idxs, decoy_mods, decoy_seq_masses

    def gen_fixed_mods(self, seq: str) -> List[ModSite]:
        """
        Generates the fixed modifications for the sequence, based on the
        input configuration.

        Args:
            seq (str): The peptide sequence.

        Returns:
            list of fixed ModSites.

        """
        nterm_mod = self.fixed_residues.get("nterm", None)
        mods = ([ModSite(self.unimod.get_mass(nterm_mod),
                         "nterm", nterm_mod)]
                if nterm_mod is not None else [])
        for ii, res in enumerate(seq):
            if res in self.fixed_residues:
                mod_name = self.fixed_residues[res]
                mods.append(
                    ModSite(self.unimod.get_mass(mod_name),
                            ii + 1, mod_name))

        return mods
