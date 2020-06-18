#! /usr/bin/env python3
"""
This module provides a base class to be inherited for validation and
retrieval pathways.

"""
from bisect import bisect_left
import functools
import logging
import operator
import os
import sys
from typing import (
    cast,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from pepfrag import ModSite

from . import (
    PSMContainer,
    machinelearning,
    mass_spectrum,
    packing,
    peptides,
    proteolysis,
    readers,
    spectra_readers
)
from .features import Features
from .readers import SearchEngine
from .rptmdetermine_config import DataSetConfig, RPTMDetermineConfig
from .validation_model import ValidationModel


MODEL_CACHE_FILE = 'model'


ScoreGetter = Callable[[readers.SearchResult], float]

ENGINE_SCORE_GETTER_MAP: Dict[SearchEngine, ScoreGetter] = {
    SearchEngine.Mascot: operator.attrgetter("ionscore"),
    SearchEngine.Comet: lambda r: r.scores["xcorr"]
}


def get_fdr_threshold(
    search_results: Iterable[readers.SearchResult],
    score_getter: ScoreGetter,
    fdr: float
) -> float:
    """
    Calculates the score threshold with the given FDR.

    Returns:
        The ion score threshold as a float.

    """
    topranking = {}
    for res in [r for r in search_results if r.rank == 1]:
        if res.spectrum in topranking:
            if score_getter(res) >= score_getter(topranking[res.spectrum]):
                topranking[res.spectrum] = res
        else:
            topranking[res.spectrum] = res

    scores = {
        readers.PeptideType.normal: [],
        readers.PeptideType.decoy: []
    }
    for res in topranking.values():
        scores[res.pep_type].append(score_getter(res))
    tscores = sorted(scores[readers.PeptideType.normal])
    dscores = sorted(scores[readers.PeptideType.decoy])

    threshold = None
    for idx, score in enumerate(tscores[::-1]):
        didx = bisect_left(dscores, score)
        dpassed = len(dscores) - didx
        if idx + dpassed == 0:
            continue
        est_fdr = dpassed / (idx + dpassed)
        if est_fdr < fdr:
            threshold = score
        if est_fdr > fdr:
            break

    if threshold is not None:
        return threshold
    raise RuntimeError(f"Failed to find score threshold at {fdr} FDR")


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


class PathwayBase:
    """
    A base class to contain common attributes and methods for validation and
    retrieval pathways for the program.

    """
    def __init__(self, config: RPTMDetermineConfig, log_file: str):
        """
        Initialize the object.

        Args:
            config:
            log_file: The name of the log file.

        """
        self.config = config

        self.modification = self.config.modification

        path_str = (f"{self.modification.replace('->', '2')}_"
                    f"{self.config.target_residue}")

        output_dir = self.config.output_dir
        if output_dir is None:
            output_dir = path_str

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
                f'Configuration unchanged since last run - using cached files'
            )

        self.proteolyzer = proteolysis.Proteolyzer(self.config.enzyme)

        # The UniMod PTM DB
        logging.info("Reading UniMod PTM DB.")
        self.ptmdb = readers.PTMDB()

        # The database search reader
        self.reader: readers.Reader = readers.get_reader(
            self.config.search_engine, self.ptmdb
        )

        self.decoy_reader: readers.Reader = readers.get_reader(
            self.config.decoy_search_engine, self.ptmdb
        )

        # Get the mass change associated with the target modification
        self.mod_mass = self.ptmdb.get_mass(self.config.modification)

        self.file_prefix = os.path.join(output_dir, f'{path_str}_')

        self.search_results: Dict[str, List[Type[readers.SearchResult]]] = {}

        self.model_features = [
            f for f in Features.all_feature_names()
            if f not in self.config.exclude_features
        ]

        self.model: Optional[ValidationModel] = None

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

    def _filter_mods(self, mods: Iterable[ModSite], seq: str)\
            -> List[ModSite]:
        """
        Filters the modification list to remove those instances of the
        target modification at the target residue.

        Args:
            mods: The peptide's ModSites.
            seq: The peptide sequence.

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

            if (mod_site.mod != self.modification and
                    seq[site - 1] != self.config.target_residue):
                new_mods.append(mod_site)

        return new_mods

    def _read_results(self):
        """
        Retrieves the search results from the set of input files.

        """
        search_results_cache = os.path.join(
            self.cache_dir, 'search_results'
        )

        if self.use_cache and os.path.exists(search_results_cache):
            logging.info('Using cached search results...')
            self.search_results = packing.load_from_file(search_results_cache)
            return

        for set_id, set_info in self.config.data_sets.items():
            res_path = os.path.join(set_info.data_dir, set_info.results)
            self.search_results[set_id] = self.reader.read(res_path)

        logging.info('Caching search results...')
        packing.save_to_file(self.search_results, search_results_cache)

    def _split_fdr(
            self,
            search_results: Sequence[readers.SearchResult],
            data_config: DataSetConfig,
    ) -> Tuple[Sequence[readers.SearchResult], Sequence[readers.SearchResult]]:
        """
        Splits the `search_results` into two lists according to the configured
        FDR confidence (for ProteinPilot) or calculated FDR threshold for other
        search engines.

        """
        positive: List[readers.SearchResult] = []
        negative: List[readers.SearchResult] = []
        if self.config.search_engine is SearchEngine.ProteinPilot:
            for res in search_results:
                (positive
                 if cast(readers.ProteinPilotSearchResult, res).confidence
                    >= data_config.confidence else negative).append(res)
            return positive, negative

        score_getter: Optional[ScoreGetter] = \
            ENGINE_SCORE_GETTER_MAP.get(self.config.search_engine)

        if score_getter is not None:
            score = get_fdr_threshold(
                search_results, score_getter, self.config.fdr)
            logging.info(f"Calculated score threshold to be {score} "
                         f"at {self.config.fdr} FDR")
            for res in search_results:
                (positive if score_getter(res) >= score
                 else negative).append(res)
            return positive, negative

        raise NotImplementedError(
            'No score getter configured for search engine '
            f'{self.config.search_engine}'
        )

    def read_mass_spectra(self) \
            -> Generator[Tuple[str, Dict[str, mass_spectrum.Spectrum]], None,
                         None]:
        """
        Reads the mass spectra from the configured spectra_files.

        Returns:


        """
        for data_conf_id, data_conf in self.config.data_sets.items():
            spectra_cache_file = os.path.join(
                self.cache_dir, f'spectra_{data_conf_id}'
            )
            if self.use_cache and os.path.exists(spectra_cache_file):
                logging.info(f'Using cached {data_conf_id} spectra...')
                spectra = packing.load_from_file(spectra_cache_file)
                yield data_conf_id, spectra
                continue

            logging.info(
                f'Processing {data_conf_id}: {data_conf.spectra_file}...'
            )
            spec_file_path = os.path.join(
                data_conf.data_dir, data_conf.spectra_file
            )

            if not os.path.isfile(spec_file_path):
                raise FileNotFoundError(
                    f"Spectra file {spec_file_path} not found"
                )

            spectra = {
                spec_id: spectrum.centroid().remove_itraq()
                for spec_id, spectrum in
                spectra_readers.read_spectra_file(spec_file_path)
            }

            logging.info(f'Caching {data_conf_id} spectra...')
            packing.save_to_file(spectra, spectra_cache_file)

            yield data_conf_id, spectra

    def _classify(self, container: PSMContainer) -> int:
        """
        Classifies all PSMs in `container`.

        Args:
            container: The PSMContainer containing PSMs to classify.

        Returns:
            The number of PSMs validated.

        """
        x = container.to_feature_array(self.model_features)
        scores = self.model.predict(x, use_cv=True)
        for psm, _scores in zip(container, scores):
            psm.ml_scores = _scores
        return machinelearning.count_consensus_votes(scores)

    def _has_target_residue(self, seq: str) -> bool:
        """
        Determines whether the given peptide `seq` contains the
        configured `target_residue`.

        Args:
            seq: The peptide sequence.

        Returns:
            Boolean indicating whether the configured residue was found in
            `seq`.

        """
        return self.config.target_residue in seq

    def _valid_peptide_length(self, seq: str) -> bool:
        """Evaluates whether the peptide `seq` is within the required length
        range."""
        return self.config.min_peptide_length <= len(seq) \
            <= self.config.max_peptide_length
