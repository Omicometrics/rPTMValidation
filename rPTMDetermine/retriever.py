#! /usr/bin/env python3
"""
A module for retrieving missed identifications from database search
results.

"""
import collections
import csv
import itertools
import os
import pickle
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import tqdm

from pepfrag import ModSite, Peptide

from .constants import DEFAULT_FRAGMENT_IONS, RESIDUES
from . import ionscore
from . import lda
from . import mass_spectrum
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from . import readers
from .retriever_config import RetrieverConfig
from . import similarity
from . import validator_base


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


def calculate_lda_probs(psms, lda_model, score_stats, features):
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
    def __init__(self, json_config):
        """
        Initialize the Retriever object.

        Args:
            json_config (json.JSON): The JSON configuration read from a file.

        """
        super().__init__(RetrieverConfig(json_config))

        print("Reading database ion scores...")
        self.db_ionscores = self._get_ionscores()

        print("Reading database search results...")
        self.db_res = self._get_search_results()

    def retrieve(self):
        """
        Retrieves missed false negative identifications from database search
        results.

        """
        print("Extracting peptide candidates...")
        peptides = self._get_peptides()
        all_spectra = self.read_mass_spectra()

        print("Building LDA validation models...")
        model, score_stats, _, features = self.build_model()
        unmod_model, unmod_score_stats, _, unmod_features = \
            self.build_unmod_model()

        spec_ids, prec_mzs = [], []
        for set_id, spectra in all_spectra.items():
            for spec_id, spec in spectra.items():
                spec_ids.append((set_id, spec_id))
                prec_mzs.append(spec.prec_mz)
        prec_mzs = np.array(prec_mzs)

        print("Finding better modified PSMs...")
        psms = self._get_better_matches(peptides, all_spectra, spec_ids,
                                        prec_mzs,
                                        tol=self.config.retrieval_tolerance)

        print("Calculating rPTMDetermine probabilities...")
        calculate_lda_probs(psms, model, score_stats, features)

        # Keep only the best match (in terms of LDA score) for each spectrum
        print("Retaining best PSM for each spectrum...")
        psms = psms.get_best_psms()

        # Attempt to correct for misassigned deamidation
        print("Attempting to correct misassigned deamidation...")
        psms = lda.apply_deamidation_correction(
            model, score_stats, psms, features, self.target_mod,
            self.proteolyzer)

        print("Deamidation removed from {} PSMs".format(
            sum(p.corrected for p in psms)))

        print("Finding unmodified analogue PSMs...")
        unmod_psms = self._find_unmod_analogues(psms)

        print("Calculating unmodified PSM features...")
        for psm in tqdm.tqdm(unmod_psms):
            psm.extract_features(None, self.proteolyzer)

        print("Calculating rPTMDetermine scores for unmodified analogues...")
        calculate_lda_probs(unmod_psms, unmod_model, unmod_score_stats,
                            unmod_features)

        unmod_psms = unmod_psms.filter_lda_prob()

        print("Calculating similarity scores...")
        psms = similarity.calculate_similarity_scores(psms, unmod_psms)

        if self.config.benchmark_file is not None:
            self.identify_benchmarks(psms)

        # Remove the identifications found by database search and validated by
        # Validate
        psms = self._remove_search_ids(psms)
        rec_psms = PSMContainer([p for p in psms if p.target])

        rec_psms = rec_psms.filter_lda_similarity(0.99,
                                                  self.config.sim_threshold)

        print("Localizing modification site(s)...")
        self._localize(rec_psms, model, features, 0.99,
                       self.config.sim_threshold)

        rec_psms = self.filter_localizations(rec_psms)

        rec_psms = rec_psms.filter_site_prob(
            self.config.site_localization_threshold)

        self.write_results(psms,
                           self.file_prefix + "all_recovered_results.csv")

        self.write_results(rec_psms,
                           self.file_prefix + "recovered_results.csv",
                           pickle_file=self.file_prefix + "rec_psms")

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

    def _get_peptides(self) \
            -> List[Tuple[str, Tuple[ModSite, ...], int, readers.PeptideType]]:
        """
        Retrieves the candidate peptides from the database search results.

        """
        allpeps: List[Tuple[str, str, str, Tuple[ModSite, ...], int,
                            readers.PeptideType]] = \
            [(set_id, spec_id, m.seq, tuple(m.mods), m.charge,
              m.pep_type)
             for set_id, spectra in self.db_res.items()
             for spec_id, matches in spectra.items()
             for m in matches]

        peps: Set[Tuple[str, Tuple[ModSite, ...], int,
                        readers.PeptideType]] = \
            {(x[2], tuple(self._filter_mods(x[3], x[2])), x[4], x[5])
             for x in allpeps if len(x[2]) >= 7 and
             set(RESIDUES).issuperset(x[2]) and
             any(r in x[2] for r in self.config.target_residues)}

        return list(peps)

    def _get_ionscores(self) \
            -> Dict[str, Dict[str, float]]:
        """
        Reads the ion scores from all databases from the configured file.

        """
        ionscores: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        with open(self.config.db_ionscores_file, 'r') as fh:
            for line in fh:
                content = line.rstrip().split()
                data_id, spec_id = content[0], content[1]
                for field in content[2:]:
                    try:
                        score = max(
                            float(s) for s in field.split(":")[1].split(",")
                            if any(field.startswith(p + ":")
                                   for p in ["ms", "pp", "ct"]))
                    except ValueError:
                        continue
                    if (spec_id not in ionscores[data_id] or
                            score > ionscores[data_id][spec_id]):
                        ionscores[data_id][spec_id] = score

        return ionscores

    def _get_better_matches(
            self,
            peptides: List[Tuple[str, Tuple[ModSite, ...], int,
                                 readers.PeptideType]],
            spectra: Dict[str, Dict[str, mass_spectrum.Spectrum]],
            spec_ids: List[Tuple[str, str]],
            prec_mzs: np.array,
            tol: float) -> PSMContainer[PSM]:
        """
        Finds the PSMs which are better than those found for other database
        search engines.

        Args:
            peptides (list): The peptide candidates.
            spectra (dict): A nested dictionary, keyed by the data set ID, then
                            the spectrum ID. Values are the mass spectra.
            spec_ids (list): A list of (Data ID, Spectrum ID) tuples.
            prec_mzs (numpy.array): The precursor mass/charge ratios.
            tol (float): The mass/charge ratio tolerance.

        Returns:
            A list of PSM objects representing those matches better than other
            matches from database search engines.

        """
        better: PSMContainer[PSM] = PSMContainer()
        for (seq, mods, charge, pep_type) in tqdm.tqdm(peptides):
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
                    modk = modj + \
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
                    by_anns = np.array([i[1] for i in by_ions])
                    by_mzs_u = by_mzs + 0.2
                    by_mzs_l = by_mzs - 0.2
                    for kk in bix:
                        try:
                            dbscore = self.db_ionscores[
                                spec_ids[kk][0]][spec_ids[kk][1]]
                        except KeyError:
                            continue

                        spec = spectra[spec_ids[kk][0]][spec_ids[kk][1]]

                        if len(spec) < 5:
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

                        score = get_ion_score(seq, charge, list(by_anns[diff]),
                                              spec[:jjm + 1, :], 0.2)

                        # Filter to those PSMs whose ion scores, before
                        # denoising, are at least close to the score from
                        # other search engines
                        if score < dbscore - 1:
                            continue

                        psm = PSM(spec_ids[kk][0], spec_ids[kk][1],
                                  mod_peptide, spectrum=spec,
                                  target=(
                                      pep_type == readers.PeptideType.normal))

                        psm.extract_features(self.config.target_mod,
                                             self.proteolyzer)
                        psm.peptide.clean_fragment_ions()

                        if round(psm.features["MatchScore"], 4) >= dbscore:
                            better.append(psm)
        return better

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
                    psm.features["MatchScore"],
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
                              data, as written during validate.py execution.

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
