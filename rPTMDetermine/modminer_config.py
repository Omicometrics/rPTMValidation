#! /usr/bin/env python3
import os
import glob
from typing import Dict, List, Optional

from config import Config, ConfigField
from readers import SearchEngine


class ModMinerConfig(Config):
    """
    This class represents the configuration options for rPTMDetermine. Its
    purpose is to centralize the possible options and their corresponding
    default values, if any.

    """
    _search_engine_field = ConfigField('search_engine',
                                       False,
                                       None,
                                       lambda v: SearchEngine[v])
    config_fields: List[ConfigField] = [
        _search_engine_field,
        ConfigField('path', False, None),
        ConfigField('mass_spec_path', False, None),
        ConfigField('model_search_res_path', False, None),
        ConfigField('files', False, None),
        ConfigField('mass_spec_files', False, None),
        ConfigField('model_search_res_files', False, None),
        ConfigField('enzyme', True, 'Trypsin'),
        ConfigField('output_path', True, 'result'),
        ConfigField('fdr', True, 0.01),
        ConfigField('log_level', True, 'INFO'),
        ConfigField('min_peptide_length', True, 7),
        ConfigField('max_peptide_length', True, 30),
        ConfigField('retrieval_tolerance', True, 0.05),
        ConfigField('num_cores', True, int(os.cpu_count() / 2)),
        ConfigField('model_search_res_engine',
                    True,
                    _search_engine_field,
                    lambda v: SearchEngine[v])
    ]

    # Type hints for dynamic fields
    search_engine: SearchEngine
    model_search_res_engine: SearchEngine
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

    def _get_files(self):
        """ Generate list of files for reading.

        """
        def _list_files(path: str, file_str: str) -> List[str]:
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

        self.mod_res_files = _list_files(self.path, self.files)
        self.res_model_files = _list_files(
            self.model_search_res_path, self.model_search_res_files)
        self.spec_files = _list_files(
            self.mass_spec_path, self.mass_spec_files)
