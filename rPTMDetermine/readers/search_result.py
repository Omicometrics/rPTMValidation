#! /usr/bin/env python3
"""
This module provides a data class to store the information defining a database
search identification.

"""
import dataclasses
import enum
from typing import Any, Dict, List, Optional

from pepfrag import ModSite


class PeptideType(enum.Enum):
    """
    An enumeration to represent the two peptide types.

    """
    normal = enum.auto()
    decoy = enum.auto()


@dataclasses.dataclass
class SearchResult():
    """
    A data class to store information about an identification from a database
    search engine. All search engine readers should return a list of
    SearchResult objects in order to standardize the interface.

    """
    seq: str
    mods: List[ModSite]
    charge: int
    spectrum: str
    rank: int
    pep_type: PeptideType
    dataset: Optional[str] = None
    time: Optional[str] = None
    confidence: Optional[float] = None
    theor_mz: Optional[float] = None
    prec_mz: Optional[float] = None
    ionscore: Optional[float] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)
