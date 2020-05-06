#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
from bisect import bisect_left
import logging
import multiprocessing as mp
import operator
import os
from typing import (Callable, cast, Dict, Iterable, List, Optional,
                    Sequence, Set, Tuple)

import numpy as np
import tqdm

from pepfrag import Peptide

from . import (
    machinelearning,
    mass_spectrum,
    peptides,
    readers,
    utilities,
    validator_base
)
from .constants import RESIDUES
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from .readers import SearchEngine
from .rptmdetermine_config import DataSetConfig, RPTMDetermineConfig

ScoreGetter = Callable[[readers.SearchResult], float]

ENGINE_SCORE_GETTER_MAP: Dict[SearchEngine, ScoreGetter] = {
    SearchEngine.Mascot: operator.attrgetter("ionscore"),
    SearchEngine.Comet: lambda r: r.scores["xcorr"]
}


def count_matched_ions(
        peptide: Peptide,
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
    return len(
        [idx for idx, mz in zip(bisect_idxs, list(spectrum.mz))
         if (idx > 0 and mz - ion_mzs[idx - 1] <= 0.2) or
         (idx < len(ion_mzs) and ion_mzs[idx] - mz <= 0.2)])


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

    scores = {readers.PeptideType.normal: [],
              readers.PeptideType.decoy: []}
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


def get_parallel_mods(psms: PSMContainer[PSM]) -> Set[str]:
    """Finds all unique modifications present in `psms`."""
    return {mod.mod for psm in psms for mod in psm.mods if mod.mod != 'Nitro'}


class Validator(validator_base.ValidateBase):
    """
    The main rPTMDetermine class. The validate method of this class
    encompasses the main functionality of the procedure.

    """
    def __init__(self, config: RPTMDetermineConfig):
        """
        Initialize the Validate object.

        Args:
            config (ValidatorConfig): The JSON configuration read from a
                                           file.

        """
        super().__init__(config, "validate.log")

        self.search_results: Dict[str, List[readers.SearchResult]] = {}

        self.psm_containers = {
            'psms': PSMContainer(),
            'neg_psms': PSMContainer(),
            'unmod_psms': PSMContainer(),
            'decoy_psms': PSMContainer(),
        }

        # Used for multiprocessing throughout the class methods
        self.pool = mp.Pool()

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

    def validate(
            self,
            model_extra_psm_containers: Optional[Sequence[PSMContainer]] = None,
            # TODO: put this into the configuration file?
            model_features: Optional[Sequence[str]] = None
    ):
        """
        Validates the identifications in the input data files.

        Args:
            model_extra_psm_containers: Additional PSMContainers containing
                                        modified identifications.
            model_features: List of features to use in model construction. If
                            None, all of the available features will be used
                            (subject to data size requirements).

        """
        # Process the input files to extract the modification identifications
        logging.info("Reading database search identifications.")
        self._read_results()
        self._get_identifications()
        allowed_mods = get_parallel_mods(self.psms)
        self._get_unmod_identifications(allowed_mods)

        self._get_decoy_identifications(allowed_mods)

        self._process_mass_spectra()

        self._calculate_features()

        self._filter_psms(lambda psm: psm.features.ErrPepMass <= 2)

        # TODO: how to handle different model configurations, mod vs. decoy etc.
        x_pos = self._subsample_unmods(
            extra_containers=model_extra_psm_containers,
            features=model_features
        )

        x_decoy = self._subsample_decoys(x_pos, features=model_features)

        model = machinelearning.construct_model(x_pos, x_decoy)

        threshold = machinelearning.calculate_score_threshold(
            model, x_pos, x_decoy
        )

        # Since only part of the positive and decoy identifications are used
        # for model construction, predict all identifications to check whether
        # the established score criteria remain valid for all identifications,
        # i.e. to ensure that FDR does not exceed 1%.
        full_fdr = machinelearning.evaluate_fdr(
            model,
            self.unmod_psms.to_feature_array(features=model_features),
            x_decoy,
            threshold
        )
        # TODO: log this + raise some error if greater than 1?
        print(full_fdr * 100)

        validated_counts = self._classify_all(
            model, threshold, features=model_features
        )
        for label, container in self.psm_containers.items():
            print(
                f'{validated_counts[label]} out of {len(container)} {label} '
                'identifications are validated'
            )

    def localize(self):
        """
        For peptide identifications with multiple possible modification sites,
        localizes the modification site by computing site probabilities.

        """

    def _read_results(self):
        """
        Retrieves the search results from the set of input files.

        """
        for set_id, set_info in self.config.data_sets.items():
            res_path = os.path.join(set_info.data_dir, set_info.results)
            self.search_results[set_id] = self.reader.read(res_path)

    def _get_identifications(self):
        """

        """
        # Target modification identifications
        for set_id, set_info in tqdm.tqdm(self.config.data_sets.items()):
            # Apply database search FDR control to the results
            pos_idents, neg_idents = self._split_fdr(
                self.search_results[set_id],
                set_info
            )
            self.psms.extend(self._results_to_mod_psms(pos_idents, set_id))
            self.neg_psms.extend(self._results_to_mod_psms(neg_idents, set_id))

        self.psms = utilities.deduplicate(self.psms)
        self.neg_psms = utilities.deduplicate(self.neg_psms)

    def _get_unmod_identifications(self, allowed_mods: Iterable[str]):
        """
        Parses the database search results to extract identifications with
        peptides containing the residues targeted by the modification under
        validation.

        Args:
            allowed_mods: The modifications allowed to exist in the "unmodified"
                          peptides.

        """
        for set_id, set_info in tqdm.tqdm(self.config.data_sets.items()):
            idents, _ = self._split_fdr(self.search_results[set_id], set_info)

            for ident in idents:
                if not self._has_target_residue(ident.seq):
                    continue
                for mod in ident.mods:
                    if (mod.mod not in allowed_mods or
                            (isinstance(mod.site, int) and
                             ident.seq[mod.site - 1] in
                             self.config.target_residues)):
                        break
                else:
                    if self._valid_peptide_length(ident.seq):
                        self.unmod_psms.append(
                            PSM(
                                set_id,
                                ident.spectrum,
                                Peptide(ident.seq, ident.charge, ident.mods)
                            )
                        )

        self.unmod_psms = utilities.deduplicate(self.unmod_psms)

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
                   and ident.seq[ms.site - 1] == res for ms in ident.mods
                   for res in self.config.target_residues):
                psms.append(
                    PSM(
                        dataset,
                        ident.spectrum,
                        Peptide(ident.seq, ident.charge, ident.mods)
                    )
                )

        return psms

    def _process_mass_spectra(self):
        """
        Processes the input mass spectra to match to their peptides.

        """
        indices = [
            container.get_index(('data_id', 'spec_id'))
            for container in self.psm_containers.values()
        ]

        for data_id, spec_id, spectrum in self.read_mass_spectra():
            for container, index in zip(self.psm_containers.values(), indices):
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

    def _subsample_decoys(
            self,
            x_pos: np.ndarray,
            retain_fraction: float = 0.2,
            features: Optional[List[str]] = None) -> np.ndarray:
        """
        Sub-samples decoy identifications to remove those matches of lowest
        quality and bring the number of identifications closer to the positive
        set.

        """
        x_decoy = self.decoy_psms.to_feature_array(features)
        x = np.concatenate((x_pos, x_decoy), axis=0)
        min_x = x.min(axis=0)
        max_x = x.max(axis=0)
        range_x = max_x - min_x

        # min-max standardization
        x_decoy_std = (x_decoy - min_x) / range_x
        x_pos_std = (x_pos - min_x) / range_x

        n_decoy = x_decoy.shape[0]

        # pre-allocate for distances between decoys and positive features
        dists = np.empty(n_decoy)

        # normalizer
        norm_decoy = np.sqrt((x_decoy_std * x_decoy_std).sum(axis=1))
        norm_pos = np.sqrt((x_pos_std * x_pos_std).sum(axis=1))

        # calculate distances using dot product to avoid memory problem due to
        # very large size of distance matrix constructed, separate them into
        # 300 blocks
        block_size = int(n_decoy / 300) + 1
        for i in range(300):
            _ix = np.arange(
                block_size * i,
                min(block_size * (i + 1), n_decoy),
                dtype=int
            )
            # calculate distance using dot product
            dist_curr = \
                np.dot(x_decoy_std[_ix], x_pos_std.T) / \
                (norm_decoy[_ix][:, np.newaxis] * norm_pos)

            dist_curr.sort(axis=1)
            # for each decoy, get the closest 3 distances to positives,
            # and calculate their mean.
            dists[_ix] = dist_curr[:, -3:].mean(axis=1)

        sorted_ix = np.argsort(dists)[::-1]
        return x_decoy[sorted_ix[:int(n_decoy * retain_fraction)]]

    def _classify_all(
            self,
            model: machinelearning.Classifier,
            score_threshold: float,
            features: Optional[Sequence[str]] = None
    ) -> Dict[str, int]:
        val_counts: Dict[str, int] = {}
        for label, psm_container in self.psm_containers.items():
            x = psm_container.to_feature_array(features=features)
            scores = model.predict(x, use_cv=True)
            # TODO: add scores to PSMs
            val_counts[label] = machinelearning.count_above_threshold(
                scores, score_threshold
            )

        return val_counts

    def _has_target_residue(self, seq: str) -> bool:
        """
        Determines whether the given peptide `seq` contains one of the
        configured `target_residues`.

        Args:
            seq: The peptide sequence.

        Returns:
            Boolean indicating whether a configured residue was found in `seq`.

        """
        return any(res in seq for res in self.config.target_residues)

    def _valid_peptide_length(self, seq: str) -> bool:
        """Evaluates whether the peptide `seq` is within the required length
        range."""
        return self.config.min_peptide_length <= len(seq) \
            <= self.config.max_peptide_length
