#! /usr/bin/env python3
import os
from typing import Dict, List, Optional

from .config import Config, ConfigField
from .readers import SearchEngine


class DataSetConfig(Config):
    """
    `Config` subclass for the data set-dependent portions of the
    `RPTMDetermineConfig`.

    """
    config_fields: List[ConfigField] = [
        ConfigField('data_dir'),
        ConfigField('confidence', True, None),
        ConfigField('results'),
        ConfigField('spectra_file'),
        ConfigField('decoy_results', True, None)
    ]

    # Type hints for dynamic fields
    data_dir: str
    confidence: Optional[float]
    results: str
    spectra_file: str
    decoy_results: Optional[str]


class RPTMDetermineConfig(Config):
    """
    This class represents the configuration options for rPTMDetermine. Its
    purpose is to centralize the possible options and their corresponding
    default values, if any.

    """
    _search_engine_field = ConfigField(
        'search_engine',
        False,
        None,
        lambda v: SearchEngine[v]
    )
    _data_sets_field = ConfigField(
        'data_sets',
        False,
        None,
        lambda v: {key: DataSetConfig(val) for key, val in v.items()}
    )
    _modification_field = ConfigField('modification')
    _target_residue_field = ConfigField('target_residue')
    config_fields: List[ConfigField] = [
        _search_engine_field,
        ConfigField(
            'decoy_search_engine',
            True,
            _search_engine_field,
            lambda v: SearchEngine[v]
        ),
        _data_sets_field,
        ConfigField('enzyme', True, 'Trypsin'),
        _modification_field,
        _target_residue_field,
        ConfigField('output_dir', True, None),
        ConfigField('exclude_features', True, []),
        ConfigField('fdr', True, 0.01),
        ConfigField('log_level', True, 'INFO'),
        ConfigField('min_peptide_length', True, 7),
        ConfigField('max_peptide_length', True, 30),
        ConfigField('retrieval_tolerance', True, 0.05),
        ConfigField('num_cores', True, os.cpu_count())
    ]

    # Type hints for dynamic fields
    search_engine: SearchEngine
    decoy_search_engine: SearchEngine
    data_sets: Dict[str, DataSetConfig]
    enzyme: str
    modification: str
    target_residue: str
    output_dir: Optional[str]
    exclude_features: List[str]
    fdr: float
    log_level: str
    min_peptide_length: int
    max_peptide_length: int
    retrieval_tolerance: float
    num_cores: int
