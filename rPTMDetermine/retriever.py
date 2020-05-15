#! /usr/bin/env python3
"""
A module for retrieving missed identifications from database search
results.

"""
import collections
import csv
import itertools
import logging
import os
import pickle
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import tqdm

from pepfrag import ModSite, Peptide

from .constants import DEFAULT_FRAGMENT_IONS, RESIDUES
from . import ionscore
from . import lda
from . import mass_spectrum
from .peptide_spectrum_match import PSM
from .peptides import merge_seq_mods
from .psm_container import PSMContainer
from . import readers
from .retriever_config import RetrieverConfig
from . import similarity
from . import validator_base


PeptideTuple = Tuple[str, Tuple[ModSite, ...], int, readers.PeptideType]


CHARGE_LABELS = [['[+]' if cj == 0 else f'[{cj + 1}+]'
                  for cj in range(charge)]
                 for charge in range(10)]

MODEL_REMOVE_COLS = ["data_id", "seq", "spec_id", "target", "score", "prob",
                     "uid"]


def get_ion_score(seq, charge, ions, spectrum, tol):
    """
    Generate variables for input mass spectrum according to the
    assigned sequence
    """
    if not ions:
        return 0.

    # sequence coverage
    nseq = max([len([v for v in ions if ck in v])
                for ck in CHARGE_LABELS[charge]])

    return ionscore.ionscore(len(seq), spectrum.shape[0], nseq,
                             spectrum[-1][0] - spectrum[0][0], tol)


def calculate_lda_probs(
        psms: PSMContainer[PSM],
        lda_model: lda.CustomPipeline,
        score_stats: lda.ScoreStatsDict,
        features: List[str]):
    """
    Calculates the LDA probs for the PSMs, using the specified pre-trained
    LDA model and features. The probabilities are set on the PSM object
    references.

    Args:
        psms (PSMContainer): The PSMs for which to calculate LDA probs.
        lda_model (lda.LDA): The trained sklearn LDA model.
        features (list): The list of features to be used.

    """
    # Convert PSMContainer to a pandas DataFrame
    batch_size = 10000
    for ii in tqdm.tqdm(range(0, len(psms), batch_size)):
        batch_psms = psms[ii:ii + batch_size]

        psms_df = batch_psms.to_df()

        lda_scores = lda_model.decide_predict(psms_df[features])[:, 0]
        for jj, psm in enumerate(batch_psms):
            psm.lda_score = lda_scores[jj]
            psm.lda_prob = lda.calculate_prob(1, psm.lda_score, score_stats)


class Retriever(validator_base.ValidateBase):
    """
    A class for retrieving missed PTM-modified peptide identifications from
    database search results, using the non-modified analogues
    to guide the search for appropriate modified peptides.

    """
    def __init__(self, config: RetrieverConfig):
        """
        Initialize the Retriever object.

        Args:
            config (RetrieverConfig): The configuration options for retrieval.

        """
        super().__init__(config, "retrieve.log")

        logging.info("Reading database search results.")
        self.db_res = self._get_search_results()

    def retrieve(self):
        """
        Retrieves missed false negative identifications from database search
        results.

        """
        logging.info("Reading mass spectra files.")
        all_spectra = self.read_mass_spectra()

        logging.info("Extracting peptide candidates.")
        peptides, ret_times = self._get_peptides(all_spectra)
        logging.info(f"{len(peptides)} candidate peptides extracted.")

        logging.info("Building LDA validation models.")
        model, score_stats, _, features = self.build_model()
        unmod_model, unmod_score_stats, _, unmod_features = \
            self.build_unmod_model()

        spec_ids, prec_mzs = [], []
        for set_id, spectra in all_spectra.items():
            for spec_id, spec in spectra.items():
                spec_ids.append((set_id, spec_id))
                prec_mzs.append(spec.prec_mz)
        prec_mzs = np.array(prec_mzs)

        logging.info("Finding modified PSMs.")
        psms = self._get_matches(peptides, all_spectra, spec_ids, prec_mzs,
                                 ret_times,
                                 tol=self.config.retrieval_tolerance)
        logging.info(f"{len(psms)} candidate PSMs identified.")

        logging.info("Calculating rPTMDetermine probabilities.")
        calculate_lda_probs(psms, model, score_stats, features)

        # Keep only the best match (in terms of LDA score) for each spectrum
        logging.info("Retaining best PSM for each spectrum.")
        psms = psms.get_best_psms()
        logging.info(f"{len(psms)} unique spectra have candidate PSMs.")

        # Attempt to correct for misassigned deamidation
        logging.info("Attempting to correct misassigned deamidation.")
        psms = lda.apply_deamidation_correction_full(
            model, score_stats, psms, features, self.target_mod,
            self.proteolyzer)

        logging.info("Deamidation removed from {} PSMs".format(
            sum(p.corrected for p in psms)))

        logging.info("Finding unmodified analogue PSMs.")
        unmod_psms = self._find_unmod_analogues(psms)

        logging.info("Calculating unmodified PSM features.")
        for psm in tqdm.tqdm(unmod_psms):
            psm.extract_features(None, self.proteolyzer)
            psm.peptide.clean_fragment_ions()

        logging.info(
            "Calculating rPTMDetermine scores for unmodified analogues.")
        calculate_lda_probs(unmod_psms, unmod_model, unmod_score_stats,
                            unmod_features)

        unmod_psms = unmod_psms.filter_lda_prob()

        logging.info("Calculating similarity scores.")
        psms = similarity.calculate_similarity_scores(psms, unmod_psms)

        if self.config.benchmark_file is not None:
            self.identify_benchmarks(psms)

        # Remove the identifications found by database search and validated by
        # Validator
        psms = self._remove_search_ids(psms)
        rec_psms = PSMContainer([p for p in psms if p.target])

        rec_psms = rec_psms.filter_lda_similarity(0.99,
                                                  self.config.sim_threshold)

        logging.info("Localizing modification site(s).")
        self._localize(rec_psms, model, features, 0.99,
                       self.config.sim_threshold)

        rec_psms = self.filter_localizations(rec_psms)

        rec_psms = rec_psms.filter_site_prob(
            self.config.site_localization_threshold)

        logging.info("Writing retrieval results to files.")

        self.write_results(psms,
                           self.file_prefix + "all_recovered_results.csv",
                           pickle_file=self.file_prefix + "all_rec_psms.pkl")

        self.write_results(rec_psms,
                           self.file_prefix + "recovered_results.csv",
                           pickle_file=self.file_prefix + "rec_psms.pkl")

        logging.info("Finished retrieving identifications.")

    def _get_search_results(self) \
            -> Dict[str, Dict[str, List[readers.SearchResult]]]:
        """
        Reads the database search results files to extract identifications.

        """
        db_res: Dict[str, Dict[str, List[readers.SearchResult]]] =\
            collections.defaultdict(lambda: collections.defaultdict(list))

        for set_id, set_info in self.config.data_sets.items():
            res_path = os.path.join(set_info["data_dir"], set_info["results"])

            identifications: Sequence[readers.SearchResult] = \
                self.reader.read(res_path)

            for ident in identifications:
                db_res[set_id][ident.spectrum].append(ident)

        return db_res

    def _get_peptides(
        self,
        all_spectra: Dict[str, Dict[str, mass_spectrum.Spectrum]]) \
            -> Tuple[List[PeptideTuple],
                     Dict[PeptideTuple, Dict[str, Dict[str, float]]]]:
        """
        Retrieves the candidate peptides from the database search results.

        """
        allpeps: List[Tuple[str, str, str, Tuple[ModSite, ...], int,
                            readers.PeptideType, Optional[float]]] = \
            [(set_id, spec_id, m.seq, tuple(m.mods), m.charge,
              m.pep_type, all_spectra[set_id][spec_id].retention_time)
             for set_id, spectra in self.db_res.items()
             for spec_id, matches in spectra.items()
             for m in matches]

        residue_set = set(RESIDUES)

        # Deduplicate peptides based on sequence, charge, modifications and
        # type, whilst also retaining the retention time for later use
        peps = []
        seen: Set[PeptideTuple] = set()
        seen_add = seen.add
        retention_times: Dict[PeptideTuple,
                              Dict[str, Dict[str, List[float]]]] = \
            collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        for (set_id, spec_id, seq, mods, charge, pep_type, rt) in allpeps:
            if len(seq) >= 7 and residue_set.issuperset(seq) and \
                    any(r in seq for r in self.config.target_residues):
                filt_mods = tuple(self._filter_mods(mods, seq))
                key = (seq, filt_mods, charge, pep_type)
                if key not in seen:
                    seen_add(key)
                    peps.append((seq, filt_mods, charge, pep_type))
                if rt is not None:
                    experiment = spec_id.split(".")[0]
                    retention_times[key][set_id][experiment].append(rt)

        # Find the minimum retention time for each peptide, within the same
        # data set and experiment
        min_rts: Dict[PeptideTuple, Dict[str, Dict[str, float]]] = \
            collections.defaultdict(lambda: collections.defaultdict(dict))
        for peptide, datasets in retention_times.items():
            for set_id, experiments in datasets.items():
                for exp_id, rts in experiments.items():
                    rts = [r for r in rts if r is not None]
                    if rts:
                        min_rts[peptide][set_id][exp_id] = min(rts)

        return peps, min_rts

    def _get_matches(
            self,
            peptides: List[PeptideTuple],
            spectra: Dict[str, Dict[str, mass_spectrum.Spectrum]],
            spec_ids: List[Tuple[str, str]],
            prec_mzs: np.array,
            ret_times: Dict[PeptideTuple, Dict[str, Dict[str, float]]],
            tol: float) -> PSMContainer[PSM]:
        """
        Finds candidate PSMs.

        Args:
            peptides (list): The peptide candidates.
            spectra (dict): A nested dictionary, keyed by the data set ID, then
                            the spectrum ID. Values are the mass spectra.
            spec_ids (list): A list of (Data ID, Spectrum ID) tuples.
            prec_mzs (numpy.array): The precursor mass/charge ratios.
            ret_times (dict): The (average) unmodified peptide retention times.
            tol (float): The mass/charge ratio tolerance.

        Returns:
            A list of PSM objects.

        """
        if self.mod_mass is None:
            logging.error("mod_mass has not been set - exiting.")
            raise RuntimeError("mod_mass is not set - exiting.")

        cands: PSMContainer[PSM] = PSMContainer()
        for unmod_peptide in tqdm.tqdm(peptides):
            (seq, mods, charge, pep_type) = unmod_peptide
            pmass = Peptide(seq, charge, mods).mass
            # Check for free (non-modified target residue)
            if mods is None:
                mix = [i for i, sk in enumerate(seq)
                       if sk in self.config.target_residues]
            else:
                mix = [i for i, sk in enumerate(seq)
                       if sk in self.config.target_residues
                       and not any(jk == i + 1 for _, jk, _ in mods
                                   if isinstance(jk, int))]
            if not mix:
                continue
            modj = [] if mods is None else list(mods)
            for nk in range(min(3, len(mix))):
                cmz = (pmass + self.mod_mass * (nk + 1)) / charge + 1.0073
                bix, = np.where((prec_mzs >= cmz - tol) &
                                (prec_mzs <= cmz + tol))
                if bix.size == 0:
                    continue

                for lx in itertools.combinations(mix, nk + 1):
                    modk: List[ModSite] = modj + \
                        [ModSite(self.mod_mass, j + 1, self.config.target_mod)
                         for j in lx]
                    mod_peptide = Peptide(seq, charge, modk)
                    by_ions = [
                        i for i in
                        mod_peptide.fragment(ion_types=DEFAULT_FRAGMENT_IONS)
                        if (i[1][0] == "y" or i[1][0] == "b")
                        and "-" not in i[1]
                    ]
                    by_mzs = np.array([i[0] for i in by_ions])
                    by_mzs_u = by_mzs + 0.2
                    by_mzs_l = by_mzs - 0.2
                    for kk in bix:
                        set_id = spec_ids[kk][0]
                        spec_id = spec_ids[kk][1]
                        spec = spectra[set_id][spec_id]

                        # Remove spectra with a small number of peaks
                        if len(spec) < 5:
                            continue

                        # Compare modified and unmodified identification
                        # retention times, persisting with the PSM only if the
                        # modified retention time is within an expected range
                        # of the unmodified peptide retention time
                        if (self.config.filter_retention_times() and
                                not self.eval_retention_time(
                                    unmod_peptide, modk, spec, set_id,
                                    spec_id, ret_times)):
                            continue

                        mzk = spec[:, 0]
                        thix = mzk.searchsorted(by_mzs_l)
                        thix2 = mzk.searchsorted(by_mzs_u)

                        diff = thix2 - thix >= 1
                        if np.count_nonzero(diff) <= 3:
                            continue

                        jjm = thix[diff].max()
                        if jjm <= 5:
                            continue

                        psm = PSM(spec_ids[kk][0], spec_ids[kk][1],
                                  mod_peptide, spectrum=spec,
                                  target=(
                                      pep_type == readers.PeptideType.normal))

                        psm.extract_features(self.config.target_mod,
                                             self.proteolyzer)
                        psm.peptide.clean_fragment_ions()

                        cands.append(psm)
        return cands

    def eval_retention_time(
            self,
            peptide: PeptideTuple,
            new_mods: List[ModSite],
            spectrum: mass_spectrum.Spectrum,
            data_id: str,
            spec_id: str,
            ret_times: Dict[PeptideTuple, Dict[str, Dict[str, float]]]) -> bool:
        """
        Evaluates whether a peptide identification passes the retention time
        criteria established in the configuration.

        Args:
            peptide (PeptideTuple): The unmodified peptide sequence,
                                    modifications, charge and type.
            new_mods (tuple): The modified peptide modifications, i.e.
                              including the target modification.
            spectrum (Spectrum): The mass spectrum for the assignment.
            data_id (str): The data ID of the mass spectrum.
            spec_id (str): The spectrum ID of the mass spectrum.
            ret_times

        Returns:
            Boolean flag indicating whether the identification passes all
            criteria.

        """
        exp_id = spec_id.split(".")[0]
        (seq, _, charge, _) = peptide
        if spectrum.retention_time is None:
            return True

        msg = (
            f"Rejecting {data_id}:{spec_id} {merge_seq_mods(seq, new_mods)}"
            f"({charge}+):")

        try:
            unmod_rt = ret_times[peptide][data_id][exp_id]
        except KeyError:
            unmod_exps = ret_times[peptide].get(data_id, {})
            if self.config.force_earlier_analogues and \
                    all(int(exp_id) < int(e) for e in unmod_exps):
                logging.debug(
                    f"{msg} experiment is earlier than "
                    "all unmodified identifications")
                return False
            if self.config.force_later_analogues and \
                    all(int(exp_id) > int(e) for e in unmod_exps):
                logging.debug(
                    f"{msg} experiment is later than "
                    "all unmodified identifications")
                return False
        else:
            unmod_rt /= 60.
            mod_rt = spectrum.retention_time / 60.
            if ((self.config.max_rt_below is not None and
                    mod_rt < unmod_rt +
                    self.config.max_rt_below) or
                    (self.config.max_rt_above is not None and
                     mod_rt > unmod_rt +
                     self.config.max_rt_above)):
                logging.debug(
                    f"{msg} retention time is {mod_rt} "
                    f"min versus unmodified {unmod_rt} min")
                return False    

        return True

    def _remove_search_ids(self, psms: PSMContainer) -> PSMContainer:
        """
        Removes the database search-identified results from the list of
        recovered PSMs.

        Args:
            psms (PSMContainer): The PSMs to clean.

        Returns:
            PSMContainer

        """
        with open(self.config.validated_ids_file, newline="") as fh:
            # TODO: use csv.DictReader
            reader = csv.reader(fh, delimiter="\t")
            # Skip the header row
            next(reader)
            results = [(row[0], row[1]) for row in reader]

        return psms.ids_not_in(results)

    def write_results(self, psms: Sequence[PSM], csv_file: str,
                      pickle_file: str = None):
        """
        Write the results to a CSV file and, optionally, a pickle file.

        """
        if pickle_file is not None:
            with open(pickle_file, "wb") as fh:
                pickle.dump(psms, fh)

        with open(csv_file, "w", newline="") as fh2:
            writer = csv.writer(fh2, delimiter="\t")
            writer.writerow(["DataID", "SpectrumID", "Sequence", "Mods",
                             "Charge", "IonScore", "rPTMDetermineScore",
                             "SimilarityScore", "SiteProbability"])
            for psm in psms:
                writer.writerow([
                    psm.data_id,
                    psm.spec_id,
                    psm.seq,
                    ",".join("{:6f}|{}|{}".format(*ms) for ms in psm.mods),
                    psm.charge,
                    psm.features.MatchScore,
                    psm.lda_score,
                    psm.max_similarity,
                    psm.site_prob
                ])

    def _build_model(self, model_file: str)\
            -> Tuple[lda.CustomPipeline, Dict[int, Tuple[float, float]],
                     float, List[str]]:
        """
        Constructs an LDA model from the feature data in model_file.

        Args:
            model_file (str): The path to a CSV file containing feature
                              data, as written during rptmdetermine_validate.py execution.

        Returns:

        """
        df = pd.read_csv(model_file, index_col=0)

        features = [f for f in df.columns.values
                    if f not in MODEL_REMOVE_COLS and
                    f not in self.config.exclude_features]

        return (*lda.lda_model(df, features), features)

    def build_model(self)\
            -> Tuple[lda.CustomPipeline, Dict[int, Tuple[float, float]],
                     float, List[str]]:
        """
        Constructs the LDA model for the modified identifications.

        """
        return self._build_model(self.config.model_file)

    def build_unmod_model(self)\
            -> Tuple[lda.CustomPipeline, Dict[int, Tuple[float, float]],
                     float, List[str]]:
        """
        Constructs the LDA model for the unmodified identifications.

        """
        return self._build_model(self.config.unmod_model_file)
