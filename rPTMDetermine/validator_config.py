#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""
import os
import sys
from typing import Any, Dict

from .base_config import BaseConfig


class ValidatorConfig(BaseConfig):
    """
    This class represents the configuration options for rPTMDetermine
    validation.

    """

    fields = [
        "uniprot_ptm_file",
        "sim_threshold_from_benchmarks",
    ]

    def __init__(self, json_config: Dict[str, Any]):
        """
        Initialize the ValidatorConfig class using the JSON configuration.

        """
        super().__init__(json_config)
        
    def __str__(self) -> str:
        """
        Implements the string conversion for the class.

        """
        string = super().__str__()
        for option in ValidatorConfig.fields:
            string += f"\t{option} = {getattr(self, option)}\n"
        return string

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

    def _check_required(self):
        """
        Checks that the required options have been set in the configuration
        file.

        This overrides the BaseConfig _check_required method to implement the
        check on sim_threshold_from_benchmarks.

        """
        super()._check_required()

        if (not self.sim_threshold_from_benchmarks and
                self.sim_threshold is None):
            print("sim_threshold must be specified when not using the "
                  "benchmark file")
            sys.exit(1)
