#! /usr/bin/env python
import abc
from typing import List

from .ptmdb import PTMDB
from .search_result import SearchResult


class Reader(metaclass=abc.ABCMeta):
    """
    An abstract base Reader to parse database search results files.

    """
    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the base reader.

        """
        self.ptmdb = ptmdb

    @abc.abstractmethod
    def read(self, filename: str, **kwargs) -> List[SearchResult]:
        """
        Reads the database search results file.

        Args:
            filename (str): The path to the database search results file.

        Returns:
            The search results as a list of SearchResults.

        """
