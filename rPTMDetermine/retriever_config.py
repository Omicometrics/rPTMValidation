#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""

from typing import Any, Dict, Optional

from .base_config import BaseConfig


class RetrieverConfig(BaseConfig):
    """
    This class represents the configuration options for rPTMDetermine
    retrieval.

    """
    def __init__(self, json_config: Dict[str, Any]):
        """
        Initialize the RetrieverConfig class using the JSON configuration.

        """
        super().__init__(json_config, extra_required=[
            "db_ionscores_file",
            "validated_ids_file",
            "model_file",
            "unmod_model_file",
            "sim_threshold"
        ])

    @property
    def retrieval_tolerance(self) -> float:
        """
        The m/z tolerance used in searching candidate peptides for a spectrum.

        """
        return self.json_config.get("retrieval_tolerance", 0.05)

    @property
    def validated_ids_file(self) -> Optional[str]:
        """
        The path to a CSV file containing the validated identifications
        obtained from using validate.Validate.

        """
        return self.json_config["validated_ids_file"]

    @property
    def model_file(self) -> str:
        """
        The file containing features with which to build an LDA model for
        retrieval. This can be created by outputting the pandas DataFrame
        used to build the model during validated.

        """
        return self.json_config["model_file"]

    @property
    def unmod_model_file(self) -> str:
        """
        The file containing features with which to build an LDA model for
        retrieval. This can be created by outputting the pandas DataFrame
        used to build the model during validated.

        """
        return self.json_config["unmod_model_file"]

    @property
    def db_ionscores_file(self) -> str:
        """
        The file containing the alternative database ion scores.

        """
        return self.json_config["db_ionscores_file"]

    @property
    def sim_threshold(self) -> float:
        """
        The threshold similarity score.

        """
        return self.json_config["sim_threshold"]