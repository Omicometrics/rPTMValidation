#! /usr/bin/env python3
"""
A module for retrieving missed identifications from database search
results.

"""
import collections
import csv
import itertools
import operator
import os
import pickle
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tqdm

from pepfrag import ModSite, Peptide

from .base_config import SearchEngine
from .constants import DEFAULT_FRAGMENT_IONS, RESIDUES
from . import ionscore
from . import lda
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


def get_search_results(data_sets: Dict[str, Dict[str, Any]],
                       unimod: readers.PTMDB, search_engine: SearchEngine)\
        -> Dict[str, Dict[str, List[validator_base.SpecMatch]]]:
    """
    Reads the database search results files to extract identifications.

    """
    db_res: Dict[str, Dict[str, List[validator_base.SpecMatch]]] =\
        collections.defaultdict(lambda: collections.defaultdict(list))
    for set_id, set_info in data_sets.items():
        res_path = os.path.join(set_info["data_dir"], set_info["results"])

        identifications: List[readers.SearchResult] =\
            readers.get_reader(search_engine, unimod).read(res_path)

        for ident in identifications:
            db_res[set_id][ident.spectrum].append(
                validator_base.SpecMatch(ident.seq, ident.mods,
                                         ident.charge, ident.confidence,
                                         ident.pep_type))

    return db_res


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
    for ii in range(0, len(psms), batch_size):
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

        self.db_ionscores = None

        self.db_res = get_search_results(self.config.data_sets, self.unimod,
                                         self.config.search_engine)

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

        print("Reading database ion scores...")
        self.db_ionscores = self._get_ionscores(spec_ids)

        print("Finding better modified PSMs...")
        psms = self._get_better_matches(peptides, all_spectra, spec_ids,
                                        prec_mzs,
                                        tol=self.config.retrieval_tolerance)
        psms = PSMContainer(psms)

        with open(self.file_prefix + "psms1", "wb") as fh:
            pickle.dump(psms, fh)

        print("Calculating rPTMDetermine probabilities...")
        calculate_lda_probs(psms, model, score_stats, features)

        with open(self.file_prefix + "psms2", "wb") as fh:
            pickle.dump(psms, fh)

        # Keep only the best match (in terms of LDA score) for each spectrum
        print("Retaining best PSM for each spectrum...")
        psms = psms.get_best_psms()

        with open(self.file_prefix + "psms3", "wb") as fh:
            pickle.dump(psms, fh)

        # Attempt to correct for misassigned deamidation
        psms = lda.apply_deamidation_correction(
            model, score_stats, psms, features, self.target_mod,
            self.proteolyzer)

        print("Deamidation removed from {} PSMs".format(
            sum(p.corrected for p in psms)))

        with open(self.file_prefix + "psms4", "wb") as fh:
            pickle.dump(psms, fh)

        print("Finding unmodified analogue PSMs...")
        unmod_psms = PSMContainer(self._find_unmod_analogues(psms))

        with open(self.file_prefix + "unmod_psms1", "wb") as fh:
            pickle.dump(unmod_psms, fh)

        print("Calculating unmodified PSM features...")
        for psm in tqdm.tqdm(unmod_psms):
            psm.extract_features(None, self.proteolyzer)

        with open(self.file_prefix + "unmod_psms2", "wb") as fh:
            pickle.dump(unmod_psms, fh)

        print("Calculating rPTMDetermine scores for unmodified analogues...")
        calculate_lda_probs(unmod_psms, unmod_model, unmod_score_stats,
                            unmod_features)

        unmod_psms = unmod_psms.filter_lda_prob()

        with open(self.file_prefix + "unmod_psms4", "wb") as fh:
            pickle.dump(unmod_psms, fh)

        print("Calculating similarity scores...")
        psms = similarity.calculate_similarity_scores(psms, unmod_psms)

        with open(self.file_prefix + "psms5", "wb") as fh:
            pickle.dump(psms, fh)

        if self.config.benchmark_file is not None:
            self.identify_benchmarks(psms)

        # Remove the identifications found by database search and validated by
        # Validate
        psms = self._remove_search_ids(psms)
        rec_psms = PSMContainer([p for p in psms if p.target])

        with open(self.file_prefix + "psms6", "wb") as fh:
            pickle.dump(psms, fh)

        rec_psms = rec_psms.filter_lda_similarity(0.99,
                                                  self.config.sim_threshold)

        with open(self.file_prefix + "rec_psms1", "wb") as fh:
            pickle.dump(rec_psms, fh)

        print("Localizing modification site(s)...")
        self._localize(rec_psms, model, features, 0.99,
                       self.config.sim_threshold)

        with open(self.file_prefix + "rec_psms2", "wb") as fh:
            pickle.dump(rec_psms, fh)

        rec_psms = self.filter_localizations(rec_psms)

        rec_psms = rec_psms.filter_site_prob(
            self.config.site_localization_threshold)

        with open(self.file_prefix + "rec_psms3", "wb") as fh:
            pickle.dump(rec_psms, fh)

        self.write_results(rec_psms,
                           self.file_prefix + "recovered_results.csv")

    def _get_peptides(self):
        """
        Retrieves the candidate peptides from the database search results.

        """
        allpeps = [(set_id, spec_id, m.seq, tuple(m.mods), m.theor_z,
                    m.pep_type)
                   for set_id, spectra in self.db_res.items()
                   for spec_id, matches in spectra.items() for m in matches]

        peps = {(x[2], x[3], x[4], x[5]) for x in allpeps if len(x[2]) >= 7
                and set(RESIDUES).issuperset(x[2]) and
                any(res in x[2] for res in self.config.target_residues)}

        peps2 = {(x[0], tuple(self._filter_mods(x[1], x[0])), x[2], x[3])
                 for x in peps}

        return list(peps2)

    def _get_ionscores(self, spec_ids):
        """
        """
        ionscores = collections.defaultdict(
            lambda: collections.defaultdict(list))
        with open(self.config.db_ionscores_file, 'r') as fh:
            for line in fh:
                xk = line.rstrip().split()
                for xj in xk[2:]:
                    if xj.startswith('ms'):
                        bk = [float(jk) for jk in xj.split(':')[1].split(',')]
                        ionscores[xk[0]][xk[1]].append(('res', 'ms', max(bk)))
                    elif xj.startswith('ct'):
                        bk = [float(jk) for jk in xj.split(':')[1].split(',')]
                        ionscores[xk[0]][xk[1]].append(('res', 'ct', max(bk)))
                    elif xj.startswith('pp'):
                        bk = [float(jk) for jk in xj.split(':')[1].split(',')]
                        ionscores[xk[0]][xk[1]].append(('res', 'pp', max(bk)))

        ionscores2 = collections.defaultdict(dict)
        for ky1, ky2 in spec_ids:
            try:
                bx = max([xk for xk in ionscores[ky1][ky2]
                          if isinstance(xk, tuple)],
                         key=operator.itemgetter(2))
            except ValueError:
                continue
            bx2 = [max(xk, key=operator.itemgetter(2))
                   for xk in ionscores[ky1][ky2] if isinstance(xk, list)]
            m = 0
            if bx2:
                if bx2[0][2] > m:
                    m = bx2[0][2]
            if bx:
                if bx[2] > m:
                    m = bx[2]

            ionscores2[ky1][ky2] = m

        return ionscores2

    def _get_better_matches(self, peptides, spectra, spec_ids, prec_mzs,
                            tol):
        """
        Finds the PSMs which are better than those found for other database
        search engines.

        Args:

        Returns:

        """
        better = []
        for pepk in tqdm.tqdm(peptides):
            seq, modx, charge, pep_type = pepk
            pmass = Peptide(seq, charge, modx).mass
            # Check for free (non-modified target residue)
            if modx is None:
                mix = [i for i, sk in enumerate(seq)
                       if sk in self.config.target_residues]
            else:
                mix = [i for i, sk in enumerate(seq)
                       if sk in self.config.target_residues
                       and not any(jk == i + 1 for _, jk, _ in modx
                                   if isinstance(jk, int))]
            if not mix:
                continue
            modj = [] if modx is None else list(modx)
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
                    all_ions =\
                        mod_peptide.fragment(ion_types=DEFAULT_FRAGMENT_IONS)
                    by_ions = [i for i in all_ions
                               if (i[1][0] == "y" or i[1][0] == "b")
                               and "-" not in i[1]]
                    by_mzs = np.array([i[0] for i in by_ions])
                    by_mzs_u = by_mzs + 0.2
                    by_mzs_l = by_mzs - 0.2
                    by_anns = np.array([i[1] for i in by_ions])
                    for kk in bix:
                        spec = spectra[spec_ids[kk][0]][spec_ids[kk][1]]

                        if len(spec) < 5:
                            continue

                        # Check whether the ionscore is close to the search
                        # engine scores
                        mzk = spec[:, 0]
                        thix = mzk.searchsorted(by_mzs_l)
                        thix2 = mzk.searchsorted(by_mzs_u)

                        diff = thix2 - thix >= 1
                        if np.count_nonzero(diff) <= 3:
                            continue

                        jjm = thix[diff].max()
                        if jjm <= 5:
                            continue

                        try:
                            dbscore = self.db_ionscores[
                                spec_ids[kk][0]][spec_ids[kk][1]]
                        except KeyError:
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
                                  target=(pep_type == "normal"))

                        psm.extract_features(self.config.target_mod,
                                             self.proteolyzer)

                        score = psm.features["MatchScore"]

                        psm.peptide.clean_fragment_ions()

                        if round(score, 4) >= dbscore:
                            better.append(psm)
        return better

    def _remove_search_ids(self, psms: PSMContainer):
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

        features: List[str] = list(df.columns.values)
        features = [f for f in features if f not in MODEL_REMOVE_COLS and
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