#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
from typing import Iterable, List, TextIO, Tuple

from ..base_config import SearchEngine

from .mascot_reader import MascotReader
from .msgfplus_reader import MSGFPlusReader
from .percolator_reader import PercolatorReader
from .protein_pilot_reader import ProteinPilotReader
from .tpp_reader import TPPReader
from .ptmdb import PTMDB


_TPP_ENGINES = {
    SearchEngine.Comet,
    SearchEngine.TPP,
    SearchEngine.XTandem
}


def get_reader(search_engine: SearchEngine, ptmdb: PTMDB):
    """
    Constructs the appropriate Reader based on the SearchEngine.

    Args:
        search_engine (SearchEngine): The SearchEngine used for database
                                      search.

    Returns:
        Reader.

    Raises:
        NotImplementedError.

    """
    if search_engine is SearchEngine.ProteinPilot:
        return ProteinPilotReader(ptmdb)
    if search_engine is SearchEngine.Mascot:
        return MascotReader(ptmdb)
    if search_engine is SearchEngine.MSGFPlus:
        return MSGFPlusReader(ptmdb)
    if search_engine is SearchEngine.Percolator:
        return PercolatorReader(ptmdb)
    if search_engine in _TPP_ENGINES:
        return TPPReader(ptmdb)
    raise NotImplementedError(
        f"Cannot read search results for engine: {search_engine}")


def read_fasta_sequences(fasta_file: TextIO) -> Iterable[Tuple[str, str]]:
    """
    Retrieves sequences from the input fasta_file.

    Args:
        fasta_file (TextIOWrapper): An open file handle to the fasta file.

    Yields:
        Sequences from the input file.

    """
    subseqs: List[str] = []
    for line in fasta_file:
        if line.startswith('>'):
            title = line.rstrip()
            if subseqs:
                yield title, ''.join(subseqs)
            subseqs = []
        else:
            subseqs.append(line.rstrip())
    if subseqs:
        yield title, ''.join(subseqs)
