#! /usr/bin/env python3
"""
This module provides a base class to be inherited for validation and
retrieval pathways.

"""

import collections
import functools
import logging
import os
import sys
from typing import Dict, Iterable, List, Tuple

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

        # Configure logging to go to a file and STDERR
        # TODO: define a logger on the class and use calls to this to log
        # messages using the appropriate logger
        logging.basicConfig(
            level=self.config.log_level,
            format="%(asctime)s [%(levelname)s]  %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, log_file)),
                logging.StreamHandler(sys.stdout)
            ])

        logging.info(f"Using configuration: {str(self.config)}")

        self.proteolyzer = proteolysis.Proteolyzer(self.config.enzyme)

        # The UniMod PTM DB
        logging.info("Reading UniMod PTM DB.")
        self.unimod = readers.PTMDB()

        # The database search reader
        self.reader: readers.Reader = readers.get_reader(
            self.config.search_engine, self.unimod)

        self.decoy_reader: readers.Reader = readers.get_reader(
            self.config.decoy_search_engine, self.unimod
        )

        # All database search results
        self.db_res: Dict[str, Dict[str, List[readers.SearchResult]]] = \
            collections.defaultdict(lambda: collections.defaultdict(list))

        # Get the mass change associated with the target modification
        self.mod_mass = self.unimod.get_mass(self.config.modification)

        self.file_prefix = f"{output_dir}/{path_str}_"

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

    def read_mass_spectra(self) -> \
            Dict[str, Dict[str, mass_spectrum.Spectrum]]:
        """
        Reads the mass spectra from the configured spectra_files.

        Returns:
            Dictionary mapping data set ID to spectra (dict).

        """
        for data_conf_id, data_conf in self.config.data_sets.items():
            # TODO: only one spectra file per results file (data config section)
            for spec_file in data_conf.spectra_files:
                logging.info(f'Processing {data_conf_id}: {spec_file}')
                spec_file_path = os.path.join(data_conf.data_dir, spec_file)

                if not os.path.isfile(spec_file_path):
                    raise FileNotFoundError(
                        f"Spectra file {spec_file_path} not found"
                    )

                spectra = spectra_readers.read_spectra_file(spec_file_path)
                for spec_id, spectrum in spectra:
                    spectrum.centroid().remove_itraq()
                    yield data_conf_id, spec_id, spectrum
