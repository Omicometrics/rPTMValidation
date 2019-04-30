#! /usr/bin/env python3
"""
A module for retrieving missed identifications from ProteinPilot search
results.

"""
import argparse
import collections
import csv
import itertools
import json
import operator
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tqdm

from constants import RESIDUES
import ionscore
import lda
import modifications
from peptide_spectrum_match import PSM
import plots
from psm_container import PSMContainer
import readers
import similarity
import spectra_readers
import validator_base

sys.path.append("../pepfrag")
from ion_generators import IonType
from pepfrag import Peptide


CHARGE_LABELS = [['[+]' if cj == 0 else f'[{cj + 1}+]'
                  for cj in range(charge)]
                 for charge in range(10)]

FEATURES = ["Charge", "ErrPepMass", "FracIon", "FracIon20%", "FracIonInt",
            "MatchScore", "NumIonb", "NumIonb2L", "NumIony", "NumIony2L",
            "PepLen", "PepMass", "SeqTagm", "n_missed_cleavages",
            "MatchScoreMod", "TotalIntMod"]

UNMOD_FEATURES = FEATURES[:-2]

SIMILARITY_THRESHOLD = 0.56


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


def get_proteinpilot_results(data_sets, unimod, filter_conf=False):
    """
    Reads the ProteinPilot peptide summary files to extract
    identifications, without testing the confidence threshold.

    """
    pp_res = collections.defaultdict(lambda: collections.defaultdict(list))
    for set_id, set_info in data_sets.items():
        data_dir = set_info['data_dir']
        conf = set_info["confidence"]

        summary_files = [os.path.join(data_dir, f)
                         for f in os.listdir(data_dir)
                         if 'PeptideSummary' in f and f.endswith('.txt')]

        if not summary_files:
            continue

        # Apply database search FDR control to the results
        if filter_conf:
            summaries = readers.read_peptide_summary(
                summary_files[0],
                condition=lambda r, cf=conf: float(r["Conf"]) >= cf)
        else:
            summaries = readers.read_peptide_summary(
                summary_files[0])
        for summary in summaries:
            mods = modifications.preparse_mod_string(summary.mods)

            try:
                parsed_mods = modifications.parse_mods(
                    mods, unimod)
            except modifications.UnknownModificationException:
                continue

            pp_res[set_id][summary.spec].append(
                validator_base.SpecMatch(summary.seq, parsed_mods,
                                         summary.theor_z,
                                         summary.conf,
                                         "decoy" if "REVERSED" in
                                         summary.names else "normal"))

    return pp_res


def calculate_lda_scores(psms, lda_model, features):
    """
    Calculates the LDA scores for the PSMs, using the specified pre-trained
    LDA model and features. The scores are set on the PSM object references.

    Args:
        psms (PSMContainer): The PSMs for which to calculate LDA scores.
        lda_model (lda.LDA): The trained sklearn LDA model.
        features (list): The list of features to be used.

    """
    # Convert PSMContainer to a pandas DataFrame
    psms_df = psms.to_df()

    lda_scores = lda_model.decide_predict(psms_df[features])[:, 0]
    for ii, psm in enumerate(psms):
        psm.lda_score = lda_scores[ii]


class Retriever(validator_base.ValidateBase):
    """
    A class for retrieving missed PTM-modified peptide identifications from
    the search results of ProteinPilot, using the non-modified analogues
    to guide the search for appropriate modified peptides.

    """
    def __init__(self, json_config):
        """
        Initialize the Retriever object.

        Args:
            json_config (json.JSON): The JSON configuration read from a file.

        """
        super().__init__(json_config)

        self.db_ionscores = None

        self.pp_res = get_proteinpilot_results(self.config.data_sets,
                                               self.unimod)

    def retrieve(self):
        """
        """
        peptides = self._get_peptides()
        all_spectra = self._get_spectra()

        model, spd_threshold = self.build_model()
        unmod_model, unmod_spd_threshold = self.build_unmod_model()

        spec_ids, prec_mzs = [], []
        for set_id, spectra in all_spectra.items():
            for spec_id, spec in spectra.items():
                spec_ids.append((set_id, spec_id))
                prec_mzs.append(spec.prec_mz)
        prec_mzs = np.array(prec_mzs)

        print("Reading database ion scores...")
        self.db_ionscores = self._get_ionscores(spec_ids)

        print("Finding better modified PSMs...")
        psms, types = zip(
            *self._get_better_matches(peptides, all_spectra,
                                      spec_ids, prec_mzs,
                                      tol=self.config.retrieval_tolerance))
        psms = PSMContainer(psms)

        print("Calculating rPTMDetermine scores...")
        calculate_lda_scores(psms, model, FEATURES)

        # Keep only the best match (in terms of LDA score) for each spectrum
        psms = psms.get_best_psms()

        psms = lda.apply_deamidation_correction(model, psms, self.target_mod,
                                                self.proteolyzer)
        psms.clean_fragment_ions()
        print("Deamidation removed from {} PSMs".format(
                sum(p.corrected for p in psms)))

        print("Finding unmodified analogue PSMs...")
        unmod_psms = PSMContainer(self._find_unmod_analogues(psms))

        print("Calculating unmodified PSM features...")
        for psm in unmod_psms:
            psm.extract_features(None, self.proteolyzer)
            psm.clean_fragment_ions()

        print("Calculating rPTMDetermine scores for unmodified analogues...")
        calculate_lda_scores(unmod_psms, unmod_model, UNMOD_FEATURES)

        unmod_psms = unmod_psms.filter_lda_score(unmod_spd_threshold)

        print("Calculating similarity scores...")
        psms = similarity.calculate_similarity_scores(psms, unmod_psms)

        self._identify_benchmarks(psms)

        # Remove the identifications found by ProteinPilot and validated by
        # Validate
        psms = self._remove_search_ids(psms)

        rec_psms = PSMContainer([p for idx, p in psms
                                 if types[idx] == "normal"])
        # TODO: similarity score threshold
        rec_psms = rec_psms.filter_lda_similarity(spd_threshold,
                                                  SIMILARITY_THRESHOLD)

        print("Localizing modification site(s)...")
        self.localize(psms, model, FEATURES, spd_threshold,
                      SIMILARITY_THRESHOLD)

        rec_psms = rec_psms.filter_site_prob(0.99)

        self.write_results(rec_psms,
                           "validated_recovered_identifications.csv")
        self.write_results(psms, "all_recovered_identifications.csv")

    def _get_peptides(self):
        """
        """
        allpeps = [(set_id, spec_id, m.seq, tuple(m.mods), m.theor_z,
                    m.pep_type)
                   for set_id, spectra in self.pp_res.items()
                   for spec_id, matches in spectra.items() for m in matches]

        peps = {(x[2], x[3], x[4], x[5]) for x in allpeps if len(x[2]) >= 7
                and set(RESIDUES).issuperset(x[2]) and
                any(res in x[2] for res in self.config.target_residues)}

        peps2 = set()
        for x in peps:
            mods = tuple([ms for ms in x[1] if ms.mod != self.target_mod])
            peps2.add((x[0], mods, x[2], x[3]))

        return list(peps2)

    def _get_spectra(self):
        """
        """
        spectra = {}
        for set_id, data_conf in tqdm.tqdm(self.config.data_sets.items()):
            spec_file = os.path.join(data_conf['data_dir'],
                                     data_conf['spectra_file'])

            if not os.path.isfile(spec_file):
                raise FileNotFoundError(f"Spectra file {spec_file} not found")

            set_spectra = spectra_readers.read_spectra_file(spec_file)
            for _, spec in set_spectra.items():
                spec.centroid().remove_itraq()

            spectra[set_id] = set_spectra

        return spectra

    def _get_ionscores(self, spec_ids):
        """
        """
        ionscores = collections.defaultdict(
            lambda: collections.defaultdict(list))
        with open('res_ionscores.txt', 'r') as fh:
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

        ionscores = collections.defaultdict(dict)
        for ky1, ky2 in spec_ids:
            try:
                bx = max([xk for xk in ionscores[ky1][ky2]
                          if isinstance(xk, tuple)],
                         key=operator.itemgetter(2))
            except KeyError:
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

            ionscores[ky1][ky2] = m

        return ionscores

    def _build_model(self, model_file):
        """
        """
        df = pd.read_csv(model_file, index_col=0)

        features = list(df.columns.values)
        for feature in ["data_id", "seq", "spec_id", "target"]:
            features.remove(feature)

        _, df, model = lda.lda_validate(
            df, features, self.config.fisher_threshold, kfold=False)

        results = [(r[1].score, r[1].prob)
                   for r in df.iterrows() if r[1].target]

        threshold = plots.get_validation_threshold(results, 0.99)

        return model, threshold

    def build_model(self):
        """
        """
        return self._build_model(self.config.retrieval_model_file)

    def build_unmod_model(self):
        """
        """
        return self._build_model(self.config.retrieval_unmod_model_file)

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
            seq, modx, c, pep_type = pepk
            pmass = Peptide(seq, c, modx).mass
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
                cmz = (pmass + self.mod_mass * (nk + 1)) / c + 1.0073
                bix, = np.where((prec_mzs >= cmz - tol) &
                                (prec_mzs <= cmz + tol))
                if bix.size == 0:
                    continue

                for lx in itertools.combinations(mix, nk + 1):
                    modk = modj + \
                        [modifications.ModSite(self.mod_mass, j + 1,
                                               self.config.target_mod)
                         for j in lx]
                    mod_peptide = Peptide(seq, c, modk)
                    all_ions = mod_peptide.fragment(ion_types={
                        IonType.precursor: {"neutral_losses": ["H2O", "NH3"],
                                            "itraq": True},
                        IonType.imm: {},
                        IonType.b: {"neutral_losses": ["H2O", "NH3"]},
                        IonType.y: {"neutral_losses": ["H2O", "NH3"]},
                        IonType.a: {"neutral_losses": []}
                    })
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
                            dbscore = self.db_ionscores[spec_ids[kk][0]][spec_ids[kk][1]]
                        except KeyError:
                            continue

                        score = get_ion_score(seq, c, list(by_anns[diff]),
                                              spec[:jjm + 1, :], 0.2)

                        # Filter to those PSMs whose ion scores, before
                        # denoising, are at least close to the score from
                        # other search engines
                        if score < dbscore - 1:
                            continue

                        psm = PSM(spec_ids[kk][0], spec_ids[kk][1],
                                  mod_peptide, spectrum=spec)

                        psm.extract_features(self.config.target_mod,
                                             self.proteolyzer)
                        score = psm.features["MatchScore"]

                        psm.clean_fragment_ions()

                        if round(score, 4) >= dbscore:
                            better.append((psm, pep_type))
        return better

    def _remove_search_ids(self, psms):
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

    def write_results(self, psms, csv_file, pickle_file=None):
        """
        Write the results to a CSV file and, optionally, a pickle file.

        """
        if pickle_file is not None:
            with open(pickle_file, "wb") as fh:
                pickle.dump(psms, fh)

        with open(csv_file, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["DataID", "SpectrumID", "Sequence", "Mods",
                             "Charge", "IonScore", "rPTMDetermineScore",
                             "SimilarityScore"])
            for psm in psms:
                writer.writerow([
                    psm.data_id,
                    psm.spec_id,
                    psm.seq,
                    ",".join("{:6f}|{}|{}".format(*ms) for ms in psm.mods),
                    psm.charge,
                    psm.features["MatchScore"],
                    psm.lda_score,
                    psm.max_similarity
                ])
                
                
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
    """
    The main entry point for the rPTMDetermine code.

    """
    args = parse_args()
    with open(args.config) as handle:
        conf = json.load(handle)

    retriever = Retriever(conf)
    retriever.retrieve()


if __name__ == '__main__':
    main()
