#! /usr/bin/env python
"""
A module to provide a base abstract reader to unify the interface for all
database search result readers.

"""
import abc
from typing import Callable, List, Optional

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
    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> List[SearchResult]:
        """
        Reads the database search results file.

        Args:
            filename (str): The path to the database search results file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:
            The search results as a list of optionally filtered SearchResults.

        """
