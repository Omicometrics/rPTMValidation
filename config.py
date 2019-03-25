#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""

import os
from typing import Any, Dict, List


# TODO: check required options at initialization time
class Config():
    """
    This class represents the configuration options for rPTMDetermine. Its
    purpose is to centralize the possible options and their corresponding
    default values, if any.

    """
    def __init__(self, json_config: Dict) -> None:
        """
        Initialize the Config class using the JSON configuration.

        """
        self.json_config = json_config

    @property
    def data_sets(self) -> Dict[str, Dict[str, Any]]:
        """
        The map of data set IDs to their data files and confidences.

        """
        return self.json_config["data_sets"]

    @property
    def enzyme(self) -> str:
        """
        The enzyme rule to be used to theoretically digest proteins.

        """
        return self.json_config.get("enzyme", "Trypsin")

    @property
    def fisher_threshold(self) -> float:
        """
        The minimum Fisher score required for feature selection.

        """
        return self.json_config.get("fisher_score_threshold", 0.05)

    @property
    def fixed_residues(self) -> List[str]:
        """
        The amino acid residues which bear fixed modifications.

        """
        return self.json_config["fixed_residues"]

    @property
    def target_db_path(self) -> str:
        """
        The path to the target database file.

        Raises:
            FileNotFoundError

        """
        path = self.json_config["target_database"]
        if not os.path.exists(path):
            raise FileNotFoundError("Target protein sequence database file "
                                    f"not found at {path}")
        return path

    @property
    def target_mod(self) -> str:
        """
        The modification for which to validate identifications.

        """
        return self.json_config["modification"]

    @property
    def target_residues(self) -> List[str]:
        """
        The residues targeted by target_mod.

        """
        return self.json_config["target_residues"]

    @property
    def unimod_ptm_file(self) -> str:
        """
        The path to the UniMod PTM DB file.

        Raises:
            FileNotFoundError

        """
        path = self.json_config.get("unimod_ptm_file", "unimod.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"UniMod PTM file not found at {path}")
        return path

    @property
    def uniprot_ptm_file(self) -> str:
        """
        The path to the UniProt PTM list file.

        Raises:
            FileNotFoundError

        """
        path = self.json_config.get("uniprot_ptm_file", "ptmlist.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"UniProt PTM file not found at {path}")
        return path
