#! /usr/bin/env python3
"""
This module provides a data class to store the information defining a database
search identification.

"""
import dataclasses
import enum
import itertools
from typing import Optional, Tuple, Any

from pepfrag import ModSite


class PeptideType(enum.Enum):
    """
    An enumeration to represent the two peptide types.

    """
    normal = enum.auto()
    decoy = enum.auto()


class SpectrumIDType(enum.Enum):
    """
    An enumeration to represent the spectrum ID type.

    """
    native = enum.auto()
    scan = enum.auto()


@dataclasses.dataclass(eq=True, frozen=True)
class SearchResult:  # pylint: disable=too-few-public-methods
    """ A data class to store information about an identification
    from a database search engine. All search engine readers should
    return a list of SearchResult objects in order to standardize
    the interface.

    """

    __slots__ = ("seq", "mods", "charge", "spectrum", "dataset", "rank",
                 "pep_type", "theor_mz",)

    seq: str
    mods: Tuple[ModSite, ...]
    charge: int
    spectrum: str
    spectrum_id_type: SpectrumIDType
    dataset: Optional[Any]
    rank: int
    pep_type: PeptideType
    theor_mz: Optional[float]

    def __getstate__(self):
        return {
            slot: getattr(self, slot)
            for slot in itertools.chain.from_iterable(
                getattr(cls, '__slots__', tuple())
                for cls in type(self).__mro__) if hasattr(self, slot)
        }

    def __setstate__(self, state):
        # This is added as a workaround for pickling frozen dataclasses, whereby
        # unpickling will try to use `setattr` on the frozen dataclass:
        # https://bugs.python.org/issue36424
        for slot, value in state.items():
            object.__setattr__(self, slot, value)
