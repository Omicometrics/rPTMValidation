#! /usr/bin/env python3
"""
A simple module to provide access to the configuration options for
rPTMDetermine.

"""
import enum
import os
from typing import Any, Dict, List, Optional


class MissingConfigOptionException(Exception):
    """
    An exception to be thrown when a required configuration option is missing.

    """


class SearchEngine(enum.Enum):
    """
    An enumeration to represent the search engines for which the tool can be
    used.

    """
    ProteinPilot = enum.auto()
    Mascot = enum.auto()
    Comet = enum.auto()
    XTandem = enum.auto()
    TPP = enum.auto()
    MSGFPlus = enum.auto()
    Percolator = enum.auto()


class BaseConfig():
    """
    This class represents the configuration options for rPTMDetermine. Its
    purpose is to centralize the possible options and their corresponding
    default values, if any.

    """
    def __init__(self, json_config: Dict[str, Any],
                 extra_required: Optional[List[str]] = None):
        """
        Initialize the Config class using the JSON configuration.

        """
        self.json_config = json_config

        self._required = ["data_sets", "fixed_residues", "target_db_path",
                          "target_mod", "target_residues", "search_engine"]

        if extra_required is not None:
            self._required.extend(extra_required)

        self._check_required()

    @property
    def search_engine(self) -> SearchEngine:
        """
        The database search engine for which to validate reuslts.

        """
        return SearchEngine[self.json_config["search_engine"]]

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
        path = self.json_config.get(
            "unimod_ptm_file",
            os.path.join(os.path.dirname(__file__), "readers", "unimod.xml"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"UniMod PTM file not found at {path}")
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
    def exclude_features(self) -> List[str]:
        """
        The list of features to exclude from model calculations.

        """
        return self.json_config.get("exclude_features", [])

    @property
    def fdr(self) -> Optional[float]:
        """
        The false discovery rate to be applied.

        """
        return self.json_config.get("fdr", None)

    def _check_required(self):
        """
        Checks that the required options have been set in the configuration
        file.

        """
        for attr in self._required:
            try:
                getattr(self, attr)
            except KeyError:
                raise MissingConfigOptionException(
                    f"Missing required config option: {attr}")
