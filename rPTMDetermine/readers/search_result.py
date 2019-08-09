#! /usr/bin/env python3
"""
This module provides a data class to store the information defining a database
search identification.

"""
import dataclasses
import enum
from typing import Optional, Tuple

from pepfrag import ModSite


class PeptideType(enum.Enum):
    """
    An enumeration to represent the two peptide types.

    """
    normal = enum.auto()
    decoy = enum.auto()


@dataclasses.dataclass(eq=True, frozen=True)
class SearchResult():  # pylint: disable=too-few-public-methods
    """
    A data class to store information about an identification from a database
    search engine. All search engine readers should return a list of
    SearchResult objects in order to standardize the interface.

    """

    __slots__ = ("seq", "mods", "charge", "spectrum", "dataset", "rank",
                 "pep_type", "theor_mz",)

    seq: str
    mods: Tuple[ModSite, ...]
    charge: int
    spectrum: str
    dataset: Optional[str]
    rank: int
    pep_type: PeptideType
    theor_mz: Optional[float]
