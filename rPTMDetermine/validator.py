#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import itertools
import logging
import os
import re
from typing import Callable, Iterable, Optional, Sequence, Set, List, Dict

from pepfrag import Peptide

from . import localization, readers, utilities, packing, pathway_base
from .constants import RESIDUES
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from .readers import SearchResult, PeptideType
from .results import write_psm_results
from .modminer_config import ModMinerConfig
from .psm_spectrum_id_mapper import SpectrumIDMapper
from .machinelearning import ValidationModel


def get_parallel_mods(
        psms: PSMContainer[PSM],
        target_mod: str
) -> Set[str]:
    """Finds all unique modifications present in `psms`."""
    return {
        mod.mod for psm in psms for mod in psm.mods if mod.mod != target_mod
    }


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

        # The PSMContainers are stored in this manner to make it easy to add
        # additional containers while working interactively.
        self._psms = {
            'mod_psms': [],
            'pos_unmod_psms': [],
            'decoy_psms': [],
            'neg_unmod_psms': []
        }

        self._container_output_names = {
            'mod_psms': 'ModifiedPSMs',
            'pos_unmod_psms': 'UnmodifiedPositives',
            'neg_unmod_psms': 'UnmodifiedNegatives',
            'decoy_psms': 'Decoys'
        }

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
        # Process the input files to extract the modification identifications
        logging.info("Reading database search identifications for "
                     "model construction ...")
        self._read_results(self.config.res_model_files, "model")

        # Split search results
        self._get_unmodified_identifications()

        # search results type for constructing spectrum ID mapper


        # load mass spectra for feature calculation
        self._load_mass_spectra(self.unmod_psms + self.decoy_psms)

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
        # clear the cache
        self.search_results.clear()

    def _get_mod_identifications(self):
        """
        Loads modified peptide identifications and converts to
        modified PSMs
        """
        pass

    def _results_to_mod_psms(
        self,
        search_res: Iterable[SearchResult],
    ) -> List[PSM]:
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
        psms: List[PSM] = []
        allowed_ranks = {1, 2}
        # Keep track of the top-ranking identification for the spectrum, since
        # rank 2 identifications will be retained if they are only different
        # by I/L
        current_spectrum: Optional[str] = None
        top_rank_ident: Optional[SearchResult] = None
        for ident in search_res:
            if (ident.pep_type is PeptideType.decoy or
                    ident.rank not in allowed_ranks):
                # Filter to rank 1/2, target identifications for validation
                # and ignore placeholder amino acid residue identifications
                continue

            if ident.rank == 1:
                current_spectrum = ident.spectrum
                top_rank_ident = ident
            else:
                if current_spectrum != ident.spectrum:
                    current_spectrum = None
                    top_rank_ident = None
                    continue
                m = re.fullmatch(
                    # Construct the pattern by replace I and L in the top rank
                    # peptide sequence with [IL] for regex matching
                    re.sub(r'[IL]', '[IL]', top_rank_ident.seq),
                    ident.seq
                )
                if m is None:
                    # Lower rank sequence is not an I/L isoform of the top
                    # ranking sequence
                    continue

            dataset = \
                ident.dataset if ident.dataset is not None else data_id

            if self._has_target_mod(ident.mods, ident.seq):
                psms.append(
                    PSM(
                        dataset,
                        ident.spectrum,
                        Peptide(ident.seq, ident.charge, ident.mods)
                    )
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
                          Peptide(r.seq, r.charge, r.mods))
                if r.pep_type == PeptideType.decoy:
                    psm.target = False
                psms.append(psm)

        return psms

    def _load_mass_spectra(self,
                           search_results: Sequence[PSM],
                           tag: str = "unmod"):
        """
        Processes the input mass spectra to match to their peptides.

        Args:
            search_results: List of PSMs
            tag: A tag indicates that results are for model construction
                 (using unmodified PSMs) or validation (for modified PSMs).
                 For the unmodified PSMs, mass spectra of negative and decoy
                 PSMs are clear to clean the cache.

        """
        # all search results files
        search_results_file_names = set(res.data_id for res in search_results)

        # list of mass spectra files
        mass_spec_files: Dict[str, str] = {}
        for spec_file in self.config.spec_files:
            name_split = os.path.split(os.path.basename(spec_file))
            mass_spec_files[name_split[0]] = spec_file

        for data_id, spectra in self.read_mass_spectra():
            for container, index in zip(self.psm_containers.values(), indices):
                for spec_id, spectrum in spectra.items():
                    for psm_idx in index[(data_id, spec_id)]:
                        container[psm_idx].spectrum = spectrum

    def _calculate_features(self):
        """Computes features for all PSMContainers."""

        for psm_container in self.psm_containers.values():
            for psm in psm_container:
                psm.extract_features()
                psm.peptide.clean_fragment_ions()

    def _filter_psms(self, predicate: Callable[[PSM], bool]):
        """Filters PSMs using the provided `predicate`.

        Args:
            predicate: A function defining a filter condition for the PSMs.

        """
        for psm_container in self.psm_containers.values():
            psm_container[:] = [p for p in psm_container if predicate(p)]
