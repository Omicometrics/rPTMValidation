#! /usr/bin/env python3
import os
import glob
import collections
import configparser
from typing import Dict, List, Tuple, Set, Any

from config import Config, ConfigField, MissingConfigOptionException
from readers import SearchEngine
from readers import PTMDB


SPLITTHRESHOLD = collections.namedtuple(
    "SPLITTHRESHOLD", ["threshold", "engine", "msg"])


class ParamConfigurator:
    """ Parameter configurations.

    """
    config_fields: List[ConfigField] = [
        ConfigField('engine', False, None, lambda v: SearchEngine[v]),
        ConfigField('model_search_res_engine', False, None,
                    lambda v: SearchEngine[v]),
        ConfigField('path', False, None),
        ConfigField('mass_spec_path', False, None),
        ConfigField('model_search_res_path', False, None),
        ConfigField('files', False, None),
        ConfigField('mass_spec_files', False, None),
        ConfigField('model_search_res_files', False, None),
        ConfigField("peptide_prophet_prob", True, 0.95),
        ConfigField("percolator_qvalue", True, 0.01),
        ConfigField('enzyme', True, 'Trypsin'),
        ConfigField('output_path', True, 'result'),
        ConfigField('fdr', True, 0.01),
        ConfigField('log_level', True, 'INFO'),
        ConfigField('excludes', True, ''),
        ConfigField('mod_excludes', True, {}),
        ConfigField('mod_fix', True, {}),
        ConfigField('mod_list', True, {}),
        ConfigField('min_peptide_length', True, 7),
        ConfigField('max_peptide_length', True, 30),
        ConfigField('retrieval_tolerance', True, 0.05),
        ConfigField('num_cores', True, int(os.cpu_count() / 2))
    ]

    # Type hints for dynamic fields
    engine: SearchEngine
    model_search_res_engine: SearchEngine
    peptide_prophet_prob: float
    percolator_qvalue: float
    enzyme: str
    output_path: str
    path: str
    files: str
    mass_spec_path: str
    mass_spec_files: str
    model_search_res_path: str
    model_search_res_files: str
    fdr: float
    log_level: str
    min_peptide_length: int
    max_peptide_length: int
    retrieval_tolerance: float
    num_cores: int
    mod_res_files: List[str]
    res_model_files: List[str]
    spec_files: List[str]
    excludes: str
    mod_excludes: Set[Tuple[str, ...]]
    fixed_modification: str
    mod_fix: Set[Tuple[str, ...]]
    mod_list: Dict[str, Dict[str, Any]]

    def __init__(self, config: configparser.ConfigParser):
        self.required = [f.name for f in self.config_fields
                         if not f.has_default]
        self._config = config
        self._configparser_to_dict()
        self._check_required()

        # path and files
        self.mod_res_files = self._parse_file_str(self.path, self.files)
        self.res_model_files = self._parse_file_str(
            self.model_search_res_path, self.model_search_res_files)
        self.spec_files = self._parse_file_str(
            self.mass_spec_path, self.mass_spec_files)

        # modifications
        self.mod_excludes = self._parse_mod_str(self.excludes)
        self.mod_fix = self._parse_mod_str(self.fixed_modification)

        self.mod_list = self._load_mod()

        # search engines
        self._set_threshold()

    def _configparser_to_dict(self):
        """ ConfigParser to dict. """
        for sec in self._config:
            for key, val in self._config[sec].items():
                setattr(self, key, val)

    @staticmethod
    def _parse_file_str(path: str, file_str: str) -> List[str]:
        """ Parses string to get list of files

        Returns:
            List of absolute file paths

        Raise:
            Files can't be found error.
        """
        if ";" in file_str:
            return [os.path.join(path, fl) for fl in file_str.split(';')]

        if os.path.isfile(os.path.join(path, file_str)):
            return [os.path.join(path, file_str)]

        # list of multiple files with fuzzy search
        files = glob.glob(os.path.join(path, file_str))
        if files:
            return files

        raise ValueError(f"Can't find files with '{file_str}' under "
                         f"directory {path}.")

    @staticmethod
    def _parse_mod_str(modstr: str) -> Set[Tuple[str, ...]]:
        """ Parse modification string to get list of modifications
        with sites

        Returns:
            A set of modifications with sites.
        """
        if not modstr:
            return set()

        mod_info = set()
        for sub in modstr.split(";"):
            mod_info.add(tuple(sub.split("@")))
        return mod_info

    def _load_mod(self) -> Dict[str, Dict[str, Any]]:
        """ Load modifications from a file.

        """
        if not self.modification_list:
            return {}

        if not os.path.isfile(self.modification_list):
            raise FileNotFoundError(
                f"No such file or directory: {self.modification_list}")

        mods = set(open(self.modification_list, "r").read().splitlines())

        # check any modification not in Unimod Database
        unimod_db = PTMDB().get_mods()
        unknown_mods = "; ".join([m for m in mods if m not in unimod_db])
        if unknown_mods:
            raise ValueError(f"Unknown modifications: {unknown_mods}.")

        return {m: unimod_db[m] for m in mods}

    def _set_threshold(self):
        """ Sets threshold for splitting search results.

        """
        try:
            search_engine = SearchEngine[self.search_engine]
        except KeyError:
            raise NotImplementedError(
                f"Cannot recognize engine: {self.search_engine}."
            )

        if search_engine == SearchEngine.TPP:
            self.res_split = SPLITTHRESHOLD(
                self.peptide_prophet_prob,
                search_engine,
                f"TPP peptide probability {self.peptide_prophet_prob}")
        elif search_engine == SearchEngine.Percolator:
            self.res_split = SPLITTHRESHOLD(
                self.percolator_qvalue,
                search_engine,
                f"Percolator q-value {self.percolator_qvalue}")
        else:
            self.res_split = SPLITTHRESHOLD(self.fdr, search_engine, None)

    def _check_required(self):
        """
        Checks that the required options have been set in the configuration
        file.

        Raises:
            MissingConfigOptionException.

        """
        # TODO: separate the required params into sections so that
        #       the missing parameters with corresponding section
        #       can be traced.
        for param in self.required:
            for sec in self._config:
                if self.config[sec].get(param) is None:
                    raise MissingConfigOptionException(
                        f"Missing required config option: {param}")


class ModMinerConfig(Config):
    """
    This class represents the configuration options for rPTMDetermine. Its
    purpose is to centralize the possible options and their corresponding
    default values, if any.

    """
    config_fields = ParamConfigurator.config_fields
