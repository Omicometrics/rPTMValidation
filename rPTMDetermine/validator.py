#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import itertools
import logging
import os
import re
import collections
from typing import (Callable, Iterable, Optional, Sequence,
                    Set, List, Dict, Tuple, NamedTuple)
from multiprocessing import Pool

from pepfrag import Peptide as PeptideBase

from . import localization, readers, utilities, packing, pathway_base
from .constants import RESIDUES
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from .readers import SearchResult, PeptideType
from .spectra_readers import read_spectra_file
from .results import write_psm_results
from .modminer_config import ModMinerConfig
from .modification import Mod
from .machinelearning import ValidationModel


class Peptide(PeptideBase):
    def __init__(self, seq: str, charge: int, mods: Sequence[Mod], **kwargs):
        super().__init__(seq, charge, mods, **kwargs)


def get_parallel_mods(psms: Sequence[PSM], target_mod: str) -> Set[str]:
    """ Finds all unique modifications present in `psms`. """
    return {
        mod.mod for psm in psms for mod in psm.mods if mod.mod != target_mod
    }


def extract_features(psm):
    psm.extract_features()


class Validator(pathway_base.PathwayBase):
    """
    The main rPTMDetermine class. The validate method of this class
    encompasses the main functionality of the procedure.

    """
    def __init__(self, config: ModMinerConfig):
        """
        Initialize the Validator object.

        Args:
            config (ModMinerConfig): The ModMinerConfig from parameters.

        """
        super().__init__(config, "validate.log")

        # The PSMs are stored in this manner to make it easy to add
        # additional psms while working interactively.
        self._psms = {'mod_psms': [],
                      'pos_unmod_psms': [],
                      'decoy_psms': [],
                      'neg_unmod_psms': []}

        self._container_output_names = {
            'mod_psms': 'ModifiedPSMs',
            'pos_unmod_psms': 'UnmodifiedPositives',
            'neg_unmod_psms': 'UnmodifiedNegatives',
            'decoy_psms': 'Decoys'
        }

        # Set of psm IDs that retain mass spectra
        self._psm_spectrum_retains = {}

    @property
    def psms(self) -> List[PSM]:
        return (self._psms['mod_psms']
                + self._psms['pos_unmod_psms']
                + self._psms['neg_unmod_psms'])

    @psms.setter
    def psms(self, value: Sequence[PSM]):
        self._psms['psms'] = value

    @property
    def mod_psms(self) -> List[PSM]:
        return self._psms['mod_psms']

    @mod_psms.setter
    def mod_psms(self, value: Iterable[PSM]):
        self._psms['mod_psms'] = value

    @property
    def unmod_psms(self) -> List[PSM]:
        return self._psms['pos_unmod_psms'] + self._psms['neg_unmod_psms']

    @property
    def decoy_psms(self) -> List[PSM]:
        return self._psms['decoy_psms']

    @decoy_psms.setter
    def decoy_psms(self, value: Iterable[PSM]):
        self._psms['decoy_psms'] = value

    @property
    def neg_unmod_psms(self) -> List[PSM]:
        return self._psms['neg_unmod_psms']

    @neg_unmod_psms.setter
    def neg_unmod_psms(self, value: Iterable[PSM]):
        self._psms['neg_unmod_psms'] = value

    def validate(
            self,
            model_extra_psm_containers: Optional[Sequence[PSMContainer]] = None
    ):
        """
        Validates the identifications in the input data files.

        Args:
            model_extra_psm_containers: Additional PSMContainers containing
                                        modified identifications.

        """
        # process the input files to extract the modification identifications
        logging.info("Reading database search identifications for "
                     "model construction ...")
        self._read_results(self.config.res_model_files, "model")

        # split search results
        self._get_unmodified_identifications()

        # get modified peptide search results
        self._get_mod_identifications()

        # clear the cache
        self.search_results.clear()

        # search results type for constructing spectrum ID mapper
        self._create_spectrum_id_mapper(self.config.model_search_res_engine)

        # load mass spectra for feature calculation
        self._load_mass_spectra(list(itertools.chain(*self._psms.values())))

        logging.info('Calculating PSM features...')
        self._calculate_features()

        self._construct_cv_model()

        # read modified peptide identifications
        mod_res_files = set(self.config.mod_res_files).difference(
            self.config.res_model_files)
        if mod_res_files:
            logging.info("Reading modified peptide identifications ...")
            self._read_results(self.config.res_model_files, "modification")

        logging.info('Classifying identifications...')
        for label, psm_container in self.psm_containers.items():
            self._classify(psm_container)
            logging.info(
                f'{len(psm_container.get_validated())} out of '
                f'{len(psm_container)} {label} identifications are validated'
            )

        self._construct_loc_model()

        logging.info('Correcting and localizing modifications...')
        for psm in itertools.chain(
            self.psms, self.neg_psms, *(model_extra_psm_containers or [])
        ):
            localization.correct_and_localize(
                psm,
                self.modification,
                self.mod_mass,
                self.config.target_residue,
                self.loc_model,
                self.model,
                self.model_features,
                self.ptmdb
            )

        logging.info('Writing results to file...')
        self._output_results()

        logging.info('Finished validation.')
        return self.model, self.loc_model

    def _construct_cv_model(self):
        """
        Constructs the machine learning model using cross validation.

        """
        model_cache = os.path.join(
            self.cache_dir, pathway_base.MODEL_CACHE_FILE
        )

        if self.use_cache and os.path.exists(model_cache):
            logging.info('Using cached model...')
            self.model = ValidationModel.from_file(model_cache)
            return

        logging.info('Training machine learning model...')
        self.model = ValidationModel(
            model_features=self.model_features, cv=3,
            n_jobs=self.config.num_cores
        )
        self.model.fit(self.unmod_psms, self.decoy_psms, self.neg_unmod_psms)

        logging.info('Caching trained model...')
        self.model.to_file(model_cache)

    def _construct_loc_model(self):
        """
        Constructs the machine learning model without cross validation for
        localization.

        """
        model_cache = os.path.join(
            self.cache_dir, pathway_base.LOCALIZATION_MODEL_CACHE_FILE
        )

        if self.use_cache and os.path.exists(model_cache):
            logging.info('Using cached localization model...')
            self.loc_model = ValidationModel.from_file(model_cache)
            return

        logging.info('Training machine learning model for localization...')
        self.loc_model = ValidationModel(
            model_features=self.model_features, cv=None,
            n_jobs=self.config.num_cores
        )
        self.loc_model.fit(
            self.unmod_psms, self.decoy_psms, self.neg_unmod_psms
        )

        logging.info('Caching trained localization model...')
        self.loc_model.to_file(model_cache)

    def _output_results(self):
        """
        Outputs the validation results to CSV format.

        A combined Excel spreadsheet, with each category of PSM in a separate
        sheet, is generated if pandas and openpyxl are available.

        """
        if not os.path.isdir(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        for label, container in self.psm_containers.items():
            # Replace the container label with a more human-readable name
            # if it exists
            label = self._container_output_names.get(label, label)

            output_file = f'{self.file_prefix}{label}.csv'

            write_psm_results(container, output_file)

    def _get_unmodified_identifications(self):
        """
        Gets unmodified identifications for model construction and
        corrections of modifications.

        """
        # get the first ranked results
        search_results = [
            r for r in itertools.chain(*self.search_results.values())
            if r.rank == 1]

        # split search results based on FDR or probability
        pos_idents, neg_idents, decoy_idents = self._split_fdr(search_results)

        # SearchResult to PSM
        self._psms['pos_unmod_psms'].extend(
            self._results_to_unmod_psms(pos_idents,
                                        allowed_mods=self.config.mod_excludes))
        self._psms['neg_unmod_psms'].extend(
            self._results_to_unmod_psms(neg_idents,
                                        allowed_mods=self.config.mod_excludes))
        self._psms['decoy_psms'].extend(
            self._results_to_unmod_psms(decoy_idents,
                                        allowed_mods=self.config.mod_excludes))

        # Mass spectra in positive PSMs are retained.
        self._psm_spectrum_retains.update(
            (p.data_id[0], p.spec_id) for p in self._psms['pos_unmod_psms']
        )

    def _get_mod_identifications(self):
        """
        Loads modified peptide identifications and converts to
        modified PSMs
        """
        mod_idents = self._results_to_mod_psms(
            itertools.chain(*self.search_results.values()),
            self.config.mod_excludes)
        self._psms["mod_psms"].extend(mod_idents)
        self._psm_spectrum_retains.update(
            (p.data_id[0], p.spec_id) for p in mod_idents
        )

    @staticmethod
    def _results_to_mod_psms(search_res: Iterable[SearchResult],
                             allowed_mods: Optional[Set[str, str]] = None)\
            -> List[PSM]:
        """
        Converts `SearchResult`s to `PSM`s after filtering.

        Filters are applied on peptide type (keep only target identifications),
        identification rank (keep only top-ranking identification, and I/L
        isoforms at rank 2), amino acid residues (check all valid) and
        modifications (keep only those with the target modification).

        Args:
            search_res: The database search results.

        Returns:
            PSMContainer.

        """
        if allowed_mods is None:
            allowed_mods = {}

        psms: List[PSM] = []
        allowed_ranks = {1, 2}
        # Keep track of the top-ranking identification for the spectrum, since
        # rank 2 identifications will be retained if they are only different
        # by I/L
        current_spectrum: Optional[str] = None
        top_rank_ident: Optional[SearchResult] = None
        for r in search_res:
            if (r.pep_type is PeptideType.decoy
                    or r.rank not in allowed_ranks
                    or not r.mods):
                # Filter to rank 1/2, target identifications for validation
                # ignore placeholder amino acid residue identifications, and
                # filter empty modification
                continue

            # Filter unknown modification and modifications excluded
            mods = [Mod(**m.__dict__) for m in r.mods]
            if (any(m.mass is None for m in mods)
                    or all((m.mod, r.seq[m.int_site - 1]) in allowed_mods
                           for m in mods)):
                continue

            if r.rank == 1:
                current_spectrum = r.spectrum
                top_rank_ident = r
            else:
                if current_spectrum != r.spectrum:
                    current_spectrum = None
                    top_rank_ident = None
                    continue
                # Construct the pattern by replace I and L in the top rank
                # peptide sequence with [IL] for regex matching
                m = re.fullmatch(re.sub(r'[IL]', '[IL]', top_rank_ident.seq),
                                 r.seq)
                if m is None:
                    # Lower rank sequence is not an I/L isoform of the top
                    # ranking sequence
                    continue

            psms.append(
                PSM(r.dataset, r.spectrum, Peptide(r.seq, r.charge, mods))
            )

        return psms

    @staticmethod
    def _results_to_unmod_psms(
            search_res: Iterable[SearchResult],
            allowed_mods: Optional[Set[str, str]] = None
    ) -> List[PSM]:
        """
        Converts `SearchResult`s to `PSM`s after filtering for unmodified PSMs.

        Filters are applied on peptide type (keep only target identifications)
        and amino acid residues (check all valid).

        Args:
            search_res: The database search results.
            allowed_mods: The modifications which may be included in
                          "unmodified" peptide identifications.

        Returns:
            PSMContainer.

        """
        if allowed_mods is None:
            allowed_mods = {}

        psms: List[PSM] = []
        for r in search_res:
            if (not r.mods or all(
                    isinstance(m.site, int) and (m.mod, r.seq[m.site - 1])
                    in allowed_mods for m in r.mods)):
                psm = PSM(r.dataset,
                          r.spectrum,
                          Peptide(r.seq, r.charge,
                                  [Mod(**m.__dict__) for m in r.mods])
                          )
                if r.pep_type is PeptideType.decoy:
                    psm.target = False
                psms.append(psm)

        return psms

    def _load_mass_spectra(self, search_results: Sequence[PSM]):
        """
        Processes the input mass spectra to match to their peptides.

        Args:
            search_results: List of PSMs
            tag: A tag indicates that results are for model construction
                 (using unmodified PSMs) or validation (for modified PSMs).
                 For the unmodified PSMs, mass spectra of negative and decoy
                 PSMs are clear to clean the cache.

        """
        # Search results grouped by file names and spectrum IDs
        psm_collection: Dict[str, Dict[str, List[PSM]]] =\
            collections.defaultdict(lambda: collections.defaultdict(list))
        for p in search_results:
            name, _ = p.data_id
            psm_collection[name][p.spec_id].append(p)

        # Raise errors if search results files are not found in list of
        # mass spectrum files.
        self._check_file_consistency(tuple(psm_collection.keys()),
                                     tuple(self._spectrum_files.keys()))

        for name in psm_collection.keys():
            spec_file = self._spectrum_files[name]
            logging.info(f'Loading mass spectra from {spec_file} ...')

            for spec_id, spectrum in read_spectra_file(spec_file):
                corr_id = self.spectrum_id_mapper(spec_id)
                for p in psm_collection[name][corr_id]:
                    p.spectrum = spectrum

            # extract features
            self._calculate_features(
                list(itertools.chain(*psm_collection[name].values()))
            )

            # clear the spectra for PSMs not in defined list
            for spec_id in psm_collection[name].keys():
                if (name, spec_id) not in self._psm_spectrum_retains:
                    for p in psm_collection[name][spec_id]:
                        p.spectrum = None

    def _calculate_features(self, psms: Sequence[PSM]):
        """Computes features for all PSMContainers."""
        with Pool(processes=self.config.num_cores) as pool:
            pool.map(extract_features, psms)

    def _filter_psms(self, predicate: Callable[[PSM], bool]):
        """Filters PSMs using the provided `predicate`.

        Args:
            predicate: A function defining a filter condition for the PSMs.

        """
        for psm_container in self.psm_containers.values():
            psm_container[:] = [p for p in psm_container if predicate(p)]
