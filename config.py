#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""

import os
import sys
from typing import Any, Dict, List, Optional


class Config():
    """
    This class represents the configuration options for rPTMDetermine. Its
    purpose is to centralize the possible options and their corresponding
    default values, if any.

    """
    def __init__(self, json_config: Dict[str, Any]):
        """
        Initialize the Config class using the JSON configuration.

        """
        self.json_config = json_config
        
        self._required = ["data_sets", "fixed_residues", "target_db_path",
                          "target_mod", "target_residues"]

        self._check_required()

    @property
    def data_sets(self) -> Dict[str, Dict[str, float]]:
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

    @property
    def correct_deamidation(self) -> bool:
        """
        Whether or not to apply the deamidation correction.

        """
        return self.json_config.get("correct_deamidation", False)

    @property
    def benchmark_file(self) -> Optional[str]:
        """
        The file containing benchmark peptides for similarity scoring
        criterion.

        """
        return self.json_config.get("benchmark_file", None)

    @property
    def model_file(self) -> Optional[str]:
        """
        The file containing features with which to build an LDA model for
        retrieval. This can be created by outputting the pandas DataFrame
        used to build the model during validated.

        """
        return self.json_config.get("model_file", None)

    @property
    def unmod_model_file(self) -> Optional[str]:
        """
        The file containing features with which to build an LDA model for
        retrieval. This can be created by outputting the pandas DataFrame
        used to build the model during validated.

        """
        return self.json_config.get("unmod_model_file", None)

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
        return self.json_config.get("validated_ids_file", None)

    @property
    def sim_threshold_from_benchmarks(self) -> bool:
        """
        A boolean flag indicating whether benchmark identifications should be
        used to dynamically define the similarity score threshold for
        validation.

        """
        return self.json_config.get("sim_threshold_from_benchmarks", True)

    @property
    def sim_threshold(self) -> Optional[float]:
        """
        The threshold similarity score. This is required if
        sim_threshold_from_benchmarks is False.

        """
        return self.json_config.get("sim_threshold", None)

    @property
    def alternative_localization_residues(self) -> List[str]:
        """
        The alternative residues targeted by the modification, but not under
        validation (i.e. in target_residues).

        """
        return self.json_config.get("alternative_localization_residues", [])

    @property
    def site_localization_threshold(self) -> float:
        """
        The probability threshold for site localization.

        """
        return self.json_config.get("site_localization_threshold", 0.99)

    @property
    def output_dir(self) -> Optional[str]:
        """
        The directory to which to write output files.

        """
        return self.json_config.get("output_dir", None)

    @property
    def db_ionscores_file(self) -> Optional[str]:
        """
        The file containing the alternative database ion scores.

        """
        return self.json_config.get("db_ionscores_file", None)
        
    @property
    def exclude_features(self) -> List[str]:
        """
        The list of features to exclude from model calculations.

        """
        return self.json_config.get("exclude_features", [])

    def _check_required(self):
        """
        Checks that the required options have been set in the configuration
        file.

        """
        for attr in self._required:
            try:
                getattr(self, attr)
            except KeyError:
                print(f"Missing required config option: {attr}")
                sys.exit(1)

        if (not self.sim_threshold_from_benchmarks and
                self.sim_threshold is None):
            print("sim_threshold must be specified when not using the "
                  "benchmark file")
            sys.exit(1)
