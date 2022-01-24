"""
This module provides a class for parsing TPP pep.xml files.

"""

import dataclasses

from .ptmdb import PTMDB
from .tpp_reader_base import TPPBaseReader, TPPSearchResult


@dataclasses.dataclass(eq=True, frozen=True)
class CometSearchResult(TPPSearchResult):
    """ Comet search results """


class CometReader(TPPBaseReader):
    """
    Class to read an Comet pep.xml file.

    """
    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the reader.

        Args:
            ptmdb (PTMDB): The UniMod PTM database.

        """
        super().__init__(ptmdb)
