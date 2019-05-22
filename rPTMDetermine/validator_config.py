#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""

import os
from typing import Any, Dict

from .base_config import BaseConfig


class ValidatorConfig(BaseConfig):
    """
    This class represents the configuration options for rPTMDetermine
    validation.

    """
    def __init__(self, json_config: Dict[str, Any]):
        """
        Initialize the ValidatorConfig class using the JSON configuration.

        """
        super().__init__(json_config)

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

    @property
    def sim_threshold_from_benchmarks(self) -> bool:
        """
        A boolean flag indicating whether benchmark identifications should be
        used to dynamically define the similarity score threshold for
        validation.

        """
        return self.json_config.get("sim_threshold_from_benchmarks", True)
