import itertools
import logging
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pepfrag import ModSite, Peptide

from . import (
    localization,
    packing,
    pathway_base,
    readers
)
from .constants import DEFAULT_FRAGMENT_IONS, RESIDUES
from .mass_spectrum import Spectrum
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from .results import write_psm_results
from .rptmdetermine_config import RPTMDetermineConfig
from .validation_model import ValidationModel


PeptideTuple = Tuple[Peptide, readers.PeptideType]


class Retriever(pathway_base.PathwayBase):
    def __init__(
            self,
            config: RPTMDetermineConfig,
            model: Optional[ValidationModel] = None,
            loc_model: Optional[ValidationModel] = None
    ):
        """
        Initialize the Retriever object.

        Args:
            config (RPTMDetermineConfig): The RPTMDetermineConfig from JSON.

        """
        super().__init__(config, "retrieve.log")
        self.model = model
        self.loc_model = loc_model

    def retrieve(self):
        """
        """
        if self.model is None:
            if not self.use_cache:
                logging.error(
                    'Cache has been invalidated by configuration change - '
                    'exiting.'
                )
                raise RuntimeError(
                    'Cache has been invalidated by configuration change'
                )

            model_cache = os.path.join(
                self.cache_dir, pathway_base.MODEL_CACHE_FILE
            )
            if os.path.exists(model_cache):
                with open(model_cache, 'rb') as fh:
                    self.model = pickle.load(fh)
            else:
                logging.error('No cached model found - exiting.')
                raise RuntimeError('No cached model found - exiting.')

            model_cache = os.path.join(
                self.cache_dir, pathway_base.LOCALIZATION_MODEL_CACHE_FILE
            )
            if os.path.exists(model_cache):
                with open(model_cache, 'rb') as fh:
                    self.loc_model = pickle.load(fh)
            else:
                logging.error('No cached localization model found - exiting.')
                raise RuntimeError(
                    'No cached localization model found - exiting.'
                )

        self._read_results()

        logging.info('Extracting peptide candidates...')
        candidate_peptides = self._get_peptide_candidates()
        logging.info(f'{len(candidate_peptides)} candidate peptides extracted.')

        all_spectra = {
            data_id: spectra for data_id, spectra in self.read_mass_spectra()
        }

        spec_ids, prec_mzs = [], []
        for data_id, spectra in all_spectra.items():
            for spec_id, spec in spectra.items():
                spec_ids.append((data_id, spec_id))
                prec_mzs.append(spec.prec_mz)
        prec_mzs = np.array(prec_mzs)

        logging.info('Finding modified PSM candidates...')
        candidates = self._get_matches(
            candidate_peptides, all_spectra, spec_ids, prec_mzs,
            tol=self.config.retrieval_tolerance
        )
        logging.info(f'{len(candidates)} candidate PSMs identified.')

        logging.info(
            'Removing spectra assigned by ProteinPilot or validated '
            'from negatives...'
        )
        candidates = self._remove_assigned(candidates)
        logging.info(f'Retained {len(candidates)} candidate PSMs.')

        logging.info('Classifying candidate PSMs...')
        self._classify(candidates)

        logging.info('Retaining best PSM for each spectrum...')
        candidates = candidates.get_best_psms()
        logging.info(f'{len(candidates)} unique spectra have candidate PSMs.')

        logging.info('Correcting isobaric modifications and localizing...')
        for psm in candidates:
            localization.correct_and_localize(
                psm,
                self.modification,
                self.mod_mass,
                self.config.target_residue,
                self.model,
                self.loc_model,
                self.model_features,
                self.ptmdb
            )

        logging.info('Writing retrieval results to file (including decoys)...')
        write_psm_results(
            candidates, f'{self.file_prefix}Retrieved.csv'
        )

        logging.info('Finished retrieving identifications.')

    def _get_peptide_candidates(self) \
            -> List[PeptideTuple]:
        """
        Retrieves the candidate peptides from the database search results.

        """
        peptide_candidate_cache = os.path.join(
            self.cache_dir, 'peptide_candidates'
        )
        if self.use_cache and os.path.exists(peptide_candidate_cache):
            logging.info('Using cached peptide candidates...')
            return packing.load_from_file(peptide_candidate_cache)

        all_peptides: List[Tuple[str, str, str, Tuple[ModSite, ...], int,
                                 readers.PeptideType]] = \
            [(set_id, ident.spectrum, ident.seq, tuple(ident.mods),
              ident.charge, ident.pep_type)
             for set_id, results in self.search_results.items()
             for ident in results]

        residue_set = set(RESIDUES)

        # Deduplicate peptides based on sequence, charge, modifications and
        # type
        peps: Set[PeptideTuple] = set()
        for (set_id, spec_id, seq, mods, charge, pep_type) in all_peptides:
            if (len(seq) >= 7 and residue_set.issuperset(seq) and
                    self._has_target_residue(seq)):
                filtered_mods = tuple(self._filter_mods(mods, seq))
                peps.add((Peptide(seq, charge, filtered_mods), pep_type))

        candidates = list(peps)

        logging.info('Caching peptide candidates...')
        packing.save_to_file(candidates, peptide_candidate_cache)

        return candidates

    def _get_matches(
            self,
            peptides: List[PeptideTuple],
            spectra: Dict[str, Dict[str, Spectrum]],
            spec_ids: List[Tuple[str, str]],
            prec_mzs: np.array,
            tol: float
    ) -> PSMContainer:
        """
        Finds candidate PSMs.

        Args:
            peptides (list): The peptide candidates.
            spectra (dict): A nested dictionary, keyed by the data set ID, then
                            the spectrum ID. Values are the mass spectra.
            spec_ids (list): A list of (Data ID, Spectrum ID) tuples.
            prec_mzs (numpy.array): The precursor mass/charge ratios.
            tol (float): The mass/charge ratio tolerance.

        Returns:
            A list of PSM objects.

        """
        if self.mod_mass is None:
            logging.error("mod_mass has not been set - exiting.")
            raise RuntimeError("mod_mass is not set - exiting.")

        candidate_cache = os.path.join(self.cache_dir, 'psm_candidates')
        if self.use_cache and os.path.exists(candidate_cache):
            logging.info('Using cached identification candidates...')
            return packing.load_from_file(candidate_cache)

        cands = PSMContainer()
        for unmod_peptide, pep_type in peptides:
            pep_mass = unmod_peptide.mass
            seq = unmod_peptide.seq
            charge = unmod_peptide.charge
            mods = unmod_peptide.mods

            # Locate non-modified target residue
            target_indices = [
                i for i, res in enumerate(seq)
                if res == self.config.target_residue
                and not any(
                    ms.site == i + 1 for ms in mods if isinstance(ms.site, int)
                )
            ]

            if not target_indices:
                continue

            base_mods = list(mods)

            for num_target_mod in range(min(3, len(target_indices))):
                cmz = (pep_mass + self.mod_mass * (num_target_mod + 1)) / \
                      charge + 1.0073

                bix, = np.where((prec_mzs >= cmz - tol) &
                                (prec_mzs <= cmz + tol))
                if bix.size == 0:
                    continue

                for lx in itertools.combinations(
                        target_indices, num_target_mod + 1
                ):
                    new_mods: List[ModSite] = base_mods + [
                        ModSite(self.mod_mass, j + 1, self.modification)
                        for j in lx
                    ]
                    mod_peptide = Peptide(seq, charge, new_mods)
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

                        mzk = spec[:, 0]
                        thix = mzk.searchsorted(by_mzs_l)
                        thix2 = mzk.searchsorted(by_mzs_u)

                        diff = thix2 - thix >= 1
                        if np.count_nonzero(diff) <= 3:
                            continue

                        jjm = thix[diff].max()
                        if jjm <= 5:
                            continue

                        psm = PSM(
                            spec_ids[kk][0],
                            spec_ids[kk][1],
                            mod_peptide,
                            spectrum=spec,
                            target=(pep_type == readers.PeptideType.normal)
                        )

                        psm.extract_features(
                            required_features=self.model_features
                        )
                        psm.peptide.clean_fragment_ions()

                        cands.append(psm)

        logging.info('Caching identification candidates...')
        packing.save_to_file(cands, candidate_cache)

        return cands

    def _remove_assigned(self, container: PSMContainer) -> PSMContainer:
        """
        Removes PSMs whose spectra were already assigned by ProteinPilot, or
        were validated from the negative set.

        Args:
            container: The candidate PSMContainer.

        Returns:
             Filtered PSMContainer.

        """
        assigned_spectra = set()
        for set_id, set_info in self.config.data_sets.items():
            # Apply database search FDR control to the results
            pos_idents, _ = self._split_fdr(
                self.search_results[set_id],
                set_info
            )
            assigned_spectra |= {
                (set_id, ident.spectrum) for ident in pos_idents
            }

        return container.ids_not_in(assigned_spectra)
