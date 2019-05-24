#! /usr/bin/env python3
"""
This module provides a data class to store the information defining a database
search identification.

"""
import dataclasses
import enum
from typing import List, Optional

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
    time: Optional[str]
    confidence: Optional[float]
    theor_mz: Optional[float]
    prec_mz: Optional[float]
    pep_type: PeptideType
