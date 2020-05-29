#! /usr/bin/env python3
"""
This module provides a base class to be inherited for validation and
retrieval pathways.

"""

import functools
import logging
import os
import pickle
import sys
from typing import Dict, Generator, Iterable, List, Tuple

from pepfrag import ModSite

from . import (
    mass_spectrum,
    peptides,
    proteolysis,
    readers,
    spectra_readers
)
from .rptmdetermine_config import RPTMDetermineConfig


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


class ValidateBase:
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

            if (mod_site.mod != self.config.target_mod and
                    seq[site - 1] != self.config.target_residue):
                new_mods.append(mod_site)

        return new_mods

    def read_mass_spectra(self) \
            -> Generator[Tuple[str, Dict[str, mass_spectrum.Spectrum]], None,
                         None]:
        """
        Reads the mass spectra from the configured spectra_files.

        Returns:


        """
        for data_conf_id, data_conf in self.config.data_sets.items():
            spectra_cache_file = os.path.join(
                self.cache_dir, f'spectra_{data_conf_id}.pkl'
            )
            if self.use_cache and os.path.exists(spectra_cache_file):
                logging.info(f'Using cached {data_conf_id} spectra')
                with open(spectra_cache_file, 'rb') as fh:
                    spectra = pickle.load(fh)
                yield data_conf_id, spectra
                continue

            logging.info(f'Processing {data_conf_id}: {data_conf.spectra_file}')
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

            logging.info(f'Caching {data_conf_id} spectra')
            with open(spectra_cache_file, 'wb') as fh:
                pickle.dump(spectra, fh)

            yield data_conf_id, spectra
