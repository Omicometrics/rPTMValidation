#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""

from typing import Any, Dict, Optional

from rPTMDetermine.config.config import BaseConfig


class RetrieverConfig(BaseConfig):
    """
    This class represents the configuration options for rPTMDetermine
    retrieval.

    """

    fields = [
        "retrieval_tolerance",
        "validated_ids_file",
        "model_file",
        "unmod_model_file",
        "db_ionscores_file",
        "sim_threshold",
        "max_rt_below",
        "max_rt_above",
        "force_earlier_analogues",
        "force_later_analogues",
    ]

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

    def __str__(self) -> str:
        """
        Implements the string conversion for the class.

        """
        string = super().__str__()
        for option in RetrieverConfig.fields:
            string += f"\t{option} = {getattr(self, option)}\n"
        return string

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
        
    @property
    def max_rt_below(self) -> Optional[float]:
        """
        The maximum reduction in retention time allowed, comparing the
        modified peptide retention time to the unmodified peptide retention
        time, in minutes.
        
        """
        return self.json_config.get("max_rt_below", None)
        
    @property
    def max_rt_above(self) -> Optional[float]:
        """
        The maximum increase in retention time allowed, comparing the
        modified peptide retention time to the unmodified peptide retention
        time, in minutes.
        
        """
        return self.json_config.get("max_rt_above", None)

    @property
    def force_earlier_analogues(self) -> bool:
        """
        Filters the initial retrieved identifications to only those whose
        unmodified analogues occur in an earlier experiment.

        """
        return self.json_config.get("force_earlier_analogues", False)

    @property
    def force_later_analogues(self) -> bool:
        """
        Filters the initial retrieved identifications to only those whose
        unmodified analogues occur in a later experiment.

        """
        return self.json_config.get("force_later_analogues", False)

    def filter_retention_times(self) -> bool:
        """
        Evaluates whether a retention time filter should be applied to
        retrieval candidates.

        """
        return self.max_rt_below is not None or self.max_rt_above is not None
