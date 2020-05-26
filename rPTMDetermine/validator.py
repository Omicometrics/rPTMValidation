#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
from bisect import bisect_left
import csv
import functools
import itertools
import logging
import multiprocessing as mp
import operator
import os
from typing import (
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type
)

import numpy as np
import tqdm

from pepfrag import Peptide

from . import (
    localization,
    machinelearning,
    readers,
    utilities,
    validator_base
)
from .constants import RESIDUES
from .features import Features
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from .readers import SearchEngine
from .rptmdetermine_config import DataSetConfig, RPTMDetermineConfig

try:
    import pandas as pd
    import openpyxl
except ImportError:
    HAS_PANDAS_EXCEL = False
else:
    HAS_PANDAS_EXCEL = True


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


def get_parallel_mods(
        psms: PSMContainer[PSM],
        target_mod: str
) -> Set[str]:
    """Finds all unique modifications present in `psms`."""
    return {
        mod.mod for psm in psms for mod in psm.mods if mod.mod != target_mod
    }


class Validator(validator_base.ValidateBase):
    """
    The main rPTMDetermine class. The validate method of this class
    encompasses the main functionality of the procedure.

    """
    def __init__(self, config: RPTMDetermineConfig):
        """
        Initialize the Validate object.

        Args:
            config (ValidatorConfig): The JSON configuration read from a file.

        """
        super().__init__(config, "validate.log")

        self.search_results: Dict[str, List[Type[readers.SearchResult]]] = {}

        # The PSMContainers are stored in this manner to make it easy to add
        # additional containers while working interactively.
        self.psm_containers = {
            'psms': PSMContainer(),
            'neg_psms': PSMContainer(),
            'unmod_psms': PSMContainer(),
            'decoy_psms': PSMContainer(),
            'neg_unmod_psms': PSMContainer()
        }

        self._container_output_names = {
            'psms': 'Positives',
            'neg_psms': 'Negatives',
            'unmod_psms': 'UnmodifiedPositives',
            'neg_unmod_psms': 'UnmodifiedNegatives',
            'decoy_psms': 'Decoys'
        }

        # Used for multiprocessing throughout the class methods
        #self.pool = mp.Pool()

        self.model: Optional[machinelearning.Classifier] = None

    @property
    def psms(self) -> PSMContainer[PSM]:
        return self.psm_containers['psms']

    @psms.setter
    def psms(self, value: Iterable[PSM]):
        self.psm_containers['psms'] = PSMContainer(value)

    @property
    def neg_psms(self) -> PSMContainer[PSM]:
        return self.psm_containers['neg_psms']

    @neg_psms.setter
    def neg_psms(self, value: Iterable[PSM]):
        self.psm_containers['neg_psms'] = PSMContainer(value)

    @property
    def unmod_psms(self) -> PSMContainer[PSM]:
        return self.psm_containers['unmod_psms']

    @unmod_psms.setter
    def unmod_psms(self, value: Iterable[PSM]):
        self.psm_containers['unmod_psms'] = PSMContainer(value)

    @property
    def decoy_psms(self) -> PSMContainer[PSM]:
        return self.psm_containers['decoy_psms']

    @decoy_psms.setter
    def decoy_psms(self, value: Iterable[PSM]):
        self.psm_containers['decoy_psms'] = PSMContainer(value)

    @property
    def neg_unmod_psms(self) -> PSMContainer[PSM]:
        return self.psm_containers['neg_unmod_psms']

    @neg_unmod_psms.setter
    def neg_unmod_psms(self, value: Iterable[PSM]):
        self.psm_containers['neg_unmod_psms'] = PSMContainer(value)

    ########################
    # Validation
    ########################

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
        model_features = [
            f for f in Features.all_feature_names()
            if f not in self.config.exclude_features
        ]

        # Process the input files to extract the modification identifications
        logging.info("Reading database search identifications.")
        self._read_results()
        self._get_mod_identifications()
        allowed_mods = get_parallel_mods(self.psms, self.modification)
        self._get_unmod_identifications(allowed_mods)

        self._get_decoy_identifications(allowed_mods)

        self._process_mass_spectra()

        self._calculate_features()

        # TODO: how to handle different model configurations, mod vs. decoy etc.
        x_pos = self._subsample_unmods(
            extra_containers=model_extra_psm_containers,
            features=model_features
        )

        x_decoy = machinelearning.subsample_negative(
            x_pos,
            self.decoy_psms.to_feature_array(model_features)
        )

        self.model = machinelearning.construct_model(x_pos, x_decoy)

        threshold = machinelearning.calculate_score_threshold(
            self.model, x_pos, x_decoy
        )

        # Since only part of the positive and decoy identifications are used
        # for model construction, predict all identifications to check whether
        # the established score criteria remain valid for all identifications,
        # i.e. to ensure that FDR does not exceed 1%.
        def eval_fdr(use_consensus=True):
            return machinelearning.evaluate_fdr(
                self.model,
                self.unmod_psms.to_feature_array(features=model_features),
                x_decoy,
                threshold,
                use_consensus=use_consensus
            )

        consensus_fdr = eval_fdr()
        majority_fdr = eval_fdr(use_consensus=False)
        # TODO: log these
        print(f'Consensus FDR: {consensus_fdr * 100}')
        print(f'Majority FDR: {majority_fdr * 100}')

        validated_counts = self._classify_all(
            threshold, features=model_features
        )
        for label, container in self.psm_containers.items():
            print(
                f'{validated_counts[label]} out of {len(container)} {label} '
                'identifications are validated'
            )

        self._correct_and_localize(
            features=model_features,
            extra_containers=model_extra_psm_containers
        )

        self._output_results(
            threshold
        )

    def _correct_and_localize(
            self,
            features: Optional[List[str]] = None,
            extra_containers: Optional[List[PSMContainer]] = None
    ):
        """
        For peptide identifications with multiple possible modification sites,
        localizes the modification site inplace by computing site probabilities.

        """
        if extra_containers is None:
            extra_containers = []

        for psm in itertools.chain(
                self.psms, self.neg_psms, *extra_containers
        ):
            cand_psms = localization.generate_deamidation_candidates(
                psm, self.ptmdb
            )
            cand_psms.extend(
                localization.generate_alternative_nterm_candidates(
                    psm,
                    self.modification,
                    'Carbamyl',
                    self.ptmdb.get_mass('Carbamyl')
                )
            )
            for cand_psm in cand_psms:
                cand_psm.extract_features()

            # Perform correction by selecting the isoform with the highest
            # score
            feature_array = cand_psms.to_feature_array(features=features)
            isoform_scores = self.model.predict(feature_array)
            max_isoform_score_idx = np.argmax(isoform_scores)
            validation_scores = self.model.predict(
                feature_array[max_isoform_score_idx],
                use_cv=True
            )

            # Replace the modifications with the corrected ones
            psm.mods = cand_psms[max_isoform_score_idx].mods
            psm.ml_scores = validation_scores

            if any(ms.mod == self.modification for ms in psm.mods):
                localization.localize(
                    psm,
                    self.modification,
                    self.mod_mass,
                    self.config.target_residue,
                    self.model,
                    features=features
                )

    def _output_results(
            self,
            threshold: float
    ):
        """
        Outputs the validation results to CSV format.

        A combined Excel spreadsheet, with each category of PSM in a separate
        sheet, is generated if pandas and openpyxl are available.

        Args:
             threshold: The classification score threshold.

        """
        if not os.path.isdir(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        output_file_base = os.path.join(
            self.config.output_dir,
            f'{self.modification}_{self.config.target_residue}'
        )

        for label, container in self.psm_containers.items():
            # Replace the container label with a more human-readable name
            # if it exists
            label = self._container_output_names.get(label, label)

            output_file = f'{output_file_base}_{label}.csv'

            with open(output_file, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    'DataID',
                    'SpectrumID',
                    'Sequence',
                    'Charge',
                    'Modifications',
                    'PassesConsensus',
                    'PassesMajority',
                    'Localized',
                    'Scores',
                    'SiteScore',
                    'SiteProbability',
                    'SiteDiffScore'
                ])
                for psm in container:
                    writer.writerow([
                        psm.data_id,
                        psm.spec_id,
                        psm.seq,
                        psm.charge,
                        ';'.join((f'{m.mod}@{m.site}' for m in psm.mods)),
                        machinelearning.passes_consensus(
                            psm.ml_scores, threshold
                        ),
                        machinelearning.passes_majority(
                            psm.ml_scores, threshold
                        ),
                        psm.is_localized(),
                        ';'.join(map(str, psm.ml_scores[0].tolist())),
                        psm.site_score,
                        psm.site_prob,
                        psm.site_diff_score
                    ])

        if HAS_PANDAS_EXCEL:
            # Generate combined output file with each CSV as a separate sheet
            with pd.ExcelWriter(f'{output_file_base}_All.xlsx') as writer:
                for label, container in self.psm_containers.items():
                    label = self._container_output_names.get(label, label)

                    output_file = f'{output_file_base}_{label}.csv'

                    df = pd.read_csv(output_file)
                    df.to_excel(writer, sheet_name=label)

    def _read_results(self):
        """
        Retrieves the search results from the set of input files.

        """
        for set_id, set_info in self.config.data_sets.items():
            res_path = os.path.join(set_info.data_dir, set_info.results)
            self.search_results[set_id] = self.reader.read(res_path)

    def _get_identifications(
            self,
            handler: Callable[[Sequence[readers.SearchResult], str],
                              PSMContainer],
            pos_container_name: str,
            neg_container_name: str
    ):
        """
        Parses the database search results to extract identifications, filtered
        using `handler`.

        Args:
            handler: A function to process the search results from each
                     configured data set. This will be passed, in turn, the
                     positive and negative identifications, as judged by FDR
                     control.
            pos_container_name: The name of the PSMContainer on this class to
                                update with the positive identifications.
            neg_container_name: The name of the PSMContainer on this class to
                                update with the negative identifications.

        """
        for set_id, set_info in tqdm.tqdm(self.config.data_sets.items()):
            # Apply database search FDR control to the results
            pos_idents, neg_idents = self._split_fdr(
                self.search_results[set_id],
                set_info
            )
            getattr(self, pos_container_name).extend(
                handler(pos_idents, set_id)
            )
            getattr(self, neg_container_name).extend(
                handler(neg_idents, set_id)
            )

        setattr(
            self,
            pos_container_name,
            utilities.deduplicate(getattr(self, pos_container_name))
        )
        setattr(
            self,
            neg_container_name,
            utilities.deduplicate(getattr(self, neg_container_name))
        )

    def _get_mod_identifications(self):
        """

        """
        self._get_identifications(
            self._results_to_mod_psms, 'psms', 'neg_psms'
        )

    def _get_unmod_identifications(self, allowed_mods: Iterable[str]):
        """
        Parses the database search results to extract identifications with
        peptides containing the residues targeted by the modification under
        validation.

        Args:
            allowed_mods: The modifications allowed to exist in the "unmodified"
                          peptides.

        """
        # noinspection PyTypeChecker
        self._get_identifications(
            functools.partial(
                self._results_to_unmod_psms,
                allowed_mods=allowed_mods
            ),
            'unmod_psms',
            'neg_unmod_psms'
        )

    def _get_decoy_identifications(self, allowed_mods: Iterable[str]):
        """

        """
        self.decoy_psms = PSMContainer()
        for set_id, set_info in self.config.data_sets.items():
            if set_info.decoy_results is not None:
                self.decoy_psms.extend(
                    self._get_decoys_from_file(set_id, set_info, allowed_mods)
                )
            else:
                # TODO: decoys from self.search_results
                pass

    def _get_decoys_from_file(
        self,
        data_id: str,
        data_config: DataSetConfig,
        allowed_mods: Iterable[str]
    ) -> PSMContainer[PSM]:
        """
        Reads the decoy identifications from the configured `decoy_results`
        file.

        Args:
            data_id: The data set ID.
            data_config: The data set configuration

        """
        decoy_psms = PSMContainer()
        res_file = os.path.join(data_config.data_dir, data_config.decoy_results)
        for ident in self.decoy_reader.read(res_file):
            if (ident.pep_type is readers.PeptideType.decoy and
                    self._has_target_residue(ident.seq) and
                    all(mod.mod in allowed_mods for mod in ident.mods) and
                    self._valid_peptide_length(ident.seq)):
                decoy_psms.append(
                    PSM(
                        data_id,
                        ident.spectrum,
                        Peptide(ident.seq, ident.charge, ident.mods)
                    )
                )
        return decoy_psms

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

    def _results_to_mod_psms(
        self,
        search_res: Iterable[readers.SearchResult],
        data_id: str
    ) -> PSMContainer[PSM]:
        """
        Converts `SearchResult`s to `PSM`s after filtering.

        Filters are applied on peptide type (keep only target identifications),
        identification rank (keep only top-ranking identification), amino acid
        residues (check all valid) and modifications (keep only those with the
        target modification).

        Args:
            search_res: The database search results.
            data_id: The ID of the data set.

        Returns:
            PSMContainer.

        """
        psms: PSMContainer[PSM] = PSMContainer()
        for ident in search_res:
            if (ident.pep_type == readers.PeptideType.decoy or
                    ident.rank != 1 or not RESIDUES.issuperset(ident.seq)):
                # Filter to rank 1, target identifications for validation
                # and ignore placeholder amino acid residue identifications
                continue

            dataset = \
                ident.dataset if ident.dataset is not None else data_id

            if any(ms.mod == self.modification and isinstance(ms.site, int)
                   and ident.seq[ms.site - 1] == self.config.target_residue
                   for ms in ident.mods):
                psms.append(
                    PSM(
                        dataset,
                        ident.spectrum,
                        Peptide(ident.seq, ident.charge, ident.mods)
                    )
                )

        return psms

    def _results_to_unmod_psms(
        self,
        search_res: Iterable[readers.SearchResult],
        data_id: str,
        allowed_mods: Iterable[str]
    ) -> PSMContainer[PSM]:
        """
        Converts `SearchResult`s to `PSM`s after filtering for unmodified PSMs.

        Filters are applied on peptide type (keep only target identifications),
        identification rank (keep only top-ranking identification) and amino
        acid residues (check all valid).

        Args:
            search_res: The database search results.
            data_id: The ID of the data set.
            allowed_mods: The modifications which may be included in
                          "unmodified" peptide identifications.

        Returns:
            PSMContainer.

        """
        psms: PSMContainer[PSM] = PSMContainer()
        for ident in search_res:
            if not self._has_target_residue(ident.seq):
                continue
            for mod in ident.mods:
                if (mod.mod not in allowed_mods or
                        (isinstance(mod.site, int) and
                         ident.seq[mod.site - 1] ==
                         self.config.target_residue)):
                    break
            else:
                if self._valid_peptide_length(ident.seq):
                    psms.append(
                        PSM(
                            data_id,
                            ident.spectrum,
                            Peptide(ident.seq, ident.charge, ident.mods)
                        )
                    )

        return psms

    ########################
    # Retrieval
    ########################

    def retrieve(self):
        """
        """
        if self.model is None:
            raise RuntimeError('validate must be called prior to retrieve')

        # TODO: generate peptide candidates, match to spectra a la version 1

    ########################
    # Utility functions
    ########################

    def _process_mass_spectra(self):
        """
        Processes the input mass spectra to match to their peptides.

        """
        indices = [
            container.get_index(('data_id', 'spec_id'))
            for container in self.psm_containers.values()
        ]

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

    def _filter_psms(self, predicate: Callable[[PSM], bool]):
        """Filters PSMs using the provided `predicate`.

        Args:
            predicate: A function defining a filter condition for the PSMs.

        """
        for psm_container in self.psm_containers.values():
            psm_container[:] = [p for p in psm_container if predicate(p)]

    def _subsample_unmods(
            self,
            extra_containers: Optional[List[PSMContainer]] = None,
            retain_fraction: float = 0.6,
            features: Optional[List[str]] = None) -> np.ndarray:
        """
        Subsamples the unmodified peptide identifications according to
        similarity to the modified identification intensity distribution.

        """
        max_int_unmod = np.log10(
            [psm.spectrum.raw_base_peak_intensity for psm in self.unmod_psms]
        )

        base_intensities = [
            psm.spectrum.raw_base_peak_intensity for psm in self.psms
        ]
        if extra_containers is not None:
            for container in extra_containers:
                base_intensities.extend([
                    psm.spectrum.raw_base_peak_intensity for psm in container
                ])

        max_int_mod = np.log10(base_intensities)

        # Get the spectra with max intensity close to modified peptide spectra
        unmod_diff_intensity = [
            np.absolute(max_int_mod - x).min() for x in max_int_unmod
        ]

        sorted_ix = np.argsort(unmod_diff_intensity)

        return self.unmod_psms.to_feature_array(features)[
            sorted_ix[:int(sorted_ix.size * retain_fraction)]
        ]

    def _classify_all(
            self,
            score_threshold: float,
            features: Optional[Sequence[str]] = None
    ) -> Dict[str, int]:
        """
        Classifies all PSMs held by the Validator using `model`.

        Args:
            score_threshold: The score cut-off for FDR (q value) control.
            features: An optional subset of features to use for each PSM. This
                      must match the features used to train the model.

        """
        val_counts: Dict[str, int] = {}
        for label, psm_container in self.psm_containers.items():
            x = psm_container.to_feature_array(features=features)
            scores = self.model.predict(x, use_cv=True)
            val_counts[label] = machinelearning.count_consensus_votes(
                scores, score_threshold
            )
            # Add scores to PSMs
            for psm, _scores in zip(psm_container, scores):
                psm.ml_scores = _scores

        return val_counts

    def _has_target_residue(self, seq: str) -> bool:
        """
        Determines whether the given peptide `seq` contains the
        configured `target_residue`.

        Args:
            seq: The peptide sequence.

        Returns:
            Boolean indicating whether the configured residue was found in `seq`.

        """
        return self.config.target_residue in seq

    def _valid_peptide_length(self, seq: str) -> bool:
        """Evaluates whether the peptide `seq` is within the required length
        range."""
        return self.config.min_peptide_length <= len(seq) \
            <= self.config.max_peptide_length
