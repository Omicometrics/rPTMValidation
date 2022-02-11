#! /usr/bin/env python3
"""
This module provides a base class to be inherited for validation and
retrieval pathways.

"""
import functools
import logging
import os
import sys
import collections
import operator
from typing import (Dict, Generator, Iterable, List,
                    Optional, Sequence, Tuple, Type)

from modification import Mod

from . import mass_spectrum, packing, peptides, proteolysis, readers

from .spectra_readers import read_spectra_file
from .base import (ScoreGetter, ScoreGetterMap,
                   PositiveGetterMap, PositiveChecker)
from .features import Features
from .readers import SearchResult, PeptideType
from .modminer_config import ModMinerConfig
from .peptide_spectrum_match import PSM
from .machinelearning import ValidationModel


def split_res(
        results: Sequence[SearchResult],
        thr: float,
        is_positive: PositiveGetterMap
) -> Tuple[List[SearchResult], List[SearchResult], List[SearchResult]]:
    """ Groups search results into positives and negatives.

    """
    positive: List[SearchResult] = []
    negative: List[SearchResult] = []
    decoys: List[SearchResult] = []
    for res in results:
        (decoys if res.pep_type is PeptideType.decoy
         else positive if is_positive(res, thr)
        else negative).append(res)
    return positive, negative, decoys


def get_fdr_threshold(
        search_results: Iterable[SearchResult],
        score_getter: ScoreGetter,
        fdr: float
) -> float:
    """ Calculates the score threshold with the given FDR.

    Returns:
        The ion score threshold as a float.

    """
    # normal and decoy indicators
    normal, decoy = PeptideType.normal.value, PeptideType.decoy.value

    # get scores and peptide types
    top_scores = {}
    SCORE = collections.namedtuple("SCORE", ["score", "type"])
    for res in search_results:
        if res.rank != 1:
            continue

        spid = res.spectrum
        if (spid not in top_scores
                or score_getter(res) >= top_scores[spid].score):
            top_scores[spid] = SCORE(score_getter(res), res.pep_type.value)

    scores = sorted(top_scores.values(),
                    key=operator.attrgetter("score"),
                    reverse=True)

    threshold = None
    n_decoy, n_target = 0, 0
    for score in scores:
        if score.type == normal:
            n_target += 1
        else:
            if n_target > 0:
                est_fdr = n_decoy / n_target
                if est_fdr <= fdr:
                    threshold = score.score
            n_decoy += 1

    if n_decoy == 0:
        raise ValueError("No decoy identification is found.")

    if threshold is not None:
        return threshold
    raise RuntimeError(f"Failed to find score threshold at {fdr} FDR")


@functools.lru_cache(maxsize=1024)
def merge_peptide_sequence(seq: str, mods: Tuple[Mod, ...]) -> str:
    """
    Merges the modifications into the peptide sequence.

    Args:
        seq (str): The peptide sequence.
        mods (tuple): The peptide ModSites.

    Returns:
        The merged sequence as a string.

    """
    return peptides.merge_seq_mods(seq, mods)


class PathwayBase:
    """
    A base class to contain common attributes and methods for validation and
    retrieval pathways for the program.

    """

    def __init__(self, config: ModMinerConfig, log_file: str):
        """
        Initialize the object.

        Args:
            config:
            log_file: The name of the log file.

        """
        self.config = config

        output_dir = self.config.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Configure logging to go to a file and STDOUT
        logging.basicConfig(
            level=self.config.log_level,
            format="%(asctime)s [%(levelname)s]  %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, log_file)),
                logging.StreamHandler(sys.stdout)
            ])

        logging.info(f"Using configuration: {str(self.config)}")

        # Determine whether the configuration has changed since the last run and
        # thus, whether cached files may be used
        self.cache_dir = os.path.join(output_dir, '.cache')
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.use_cache = self._valid_cache()
        if self.use_cache:
            logging.info(
                'Configuration unchanged since last run - using cached files'
            )

        self.proteolyzer = proteolysis.Proteolyzer(self.config.enzyme)

        # The UniMod PTM DB
        logging.info("Reading UniMod PTM DB.")
        self.ptmdb = readers.PTMDB()

        # The database search result reader
        self.reader: readers.Reader = readers.get_reader(
            self.config.engine, self.ptmdb
        )

        # The database search result reader for model construction
        self.reader_model: readers.Reader = readers.get_reader(
            self.config.model_search_res_engine, self.ptmdb
        )

        self.search_results: Dict[List[Type[SearchResult]]] = {}

        self.model_features = Features.all_feature_names()

    def _valid_cache(self) -> bool:
        """
        Determines whether any cached files may be used, based on the
        configuration file hash.

        Returns:
            Boolean indicating whether the cache is valid.

        """
        config_hash_file = os.path.join(self.cache_dir, 'CONFIG_HASH')
        if os.path.isdir(self.cache_dir) and os.path.exists(config_hash_file):
            with open(config_hash_file) as fh:
                prev_hash = fh.read()
            if hash(self.config) == int(prev_hash):
                # The configuration file is unchanged from the previous run
                # and cached files may be safely used
                return True

        with open(config_hash_file, 'w') as fh:
            fh.write(str(hash(self.config)))

        return False

    @staticmethod
    def _filter_mods(mods: Iterable[Mod], target_mod: str) -> List[Mod]:
        """
        Filters the modification list to remove those instances of the
        target modification at the target residue.

        Args:
            mods: The peptide's ModSites.
            seq: The peptide sequence.

        Returns:
            Filtered list of ModSites.

        """
        return [m for m in mods if m.mod != target_mod]

    def _read_results(self, res_files: Sequence[str], mode: str):
        """
        Reads the search results from the set of input files.

        Args:
            res_files: List of files with full directory.
            mode: Type of the results files, {'model', 'modification'}
                  for construction of validation model or validation
                  of modifications.

        """
        search_results_cache = os.path.join(self.cache_dir, 'search_results')

        if self.use_cache and os.path.exists(search_results_cache):
            logging.info('Using cached search results...')
            self.search_results = packing.load_from_file(search_results_cache)
            return

        # reader according to search engine specified
        reader = self.reader_model if mode == "model" else self.reader
        for fl in res_files:
            logging.info(f'Load search results: {fl}')
            fl_name = os.path.basename(fl)
            self.search_results[fl_name] = reader.read(fl)

    def _split_fdr(self, search_results: Sequence[SearchResult]) \
            -> Tuple[Sequence[SearchResult],
                     Sequence[SearchResult],
                     Sequence[SearchResult]]:
        """
        Splits the `search_results` into two lists according to the
        configured FDR threshold or calculated score threshold at the
        FDR for other search engines.

        """
        splitter: Optional[PositiveChecker] = \
            PositiveGetterMap.get(self.config.search_engine)

        if splitter is not None:
            logging.info("Split search results at "
                         f"{self.config.res_split.msg}...")
            return split_res(search_results,
                             self.config.res_split.threshold,
                             splitter)

        score_getter: Optional[ScoreGetter] = \
            ScoreGetterMap.get(self.config.search_engine)

        score = get_fdr_threshold(
            search_results, score_getter, self.config.fdr)

        logging.info(f"Calculated score threshold to be {score} "
                     f"at {self.config.fdr} FDR")
        splitter = lambda r, s: score_getter(r) >= s
        return split_res(search_results, score, splitter)

    def read_mass_spectra(self):
        """
        Reads the mass spectra from the configured spectra_files.

        Returns:


        """
        for ms_file in self.config.spec_files:
            logging.info(f'Loading mass spectra from {ms_file} ...')

            spectra = {
                spec_id: spectrum.centroid().remove_itraq()
                for spec_id, spectrum in read_spectra_file(spec_file_path)
            }

    def _classify(self, psms: List[PSM]):
        """
        Classifies all PSMs in `container`.

        Args:
            container: The PSMContainer containing PSMs to classify.

        """
        scores = self.model.validate(psms)
        for psm, _scores in zip(psms, scores):
            psm.ml_scores = _scores

    def _valid_peptide_length(self, seq: str) -> bool:
        """Evaluates whether the peptide `seq` is within the required length
        range."""
        return self.config.min_peptide_length <= len(seq) \
               <= self.config.max_peptide_length
