#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
import enum
from typing import Iterable, List, TextIO, Tuple

from .mascot_reader import MascotReader
from .msgfplus_reader import MSGFPlusReader
from .percolator_reader import PercolatorReader, PercolatorTextReader
from .protein_pilot_reader import ProteinPilotReader, ProteinPilotXMLReader
from .tpp_reader import TPPReader
from .ptmdb import PTMDB


class SearchEngine(enum.Enum):
    """
    An enumeration to represent the search engines for which search results
    can be read and parsed.

    """
    ProteinPilot = enum.auto()
    ProteinPilotXML = enum.auto()
    Mascot = enum.auto()
    Comet = enum.auto()
    XTandem = enum.auto()
    TPP = enum.auto()
    MSGFPlus = enum.auto()
    Percolator = enum.auto()
    PercolatorText = enum.auto()


ENGINE_READER_MAP = {
    SearchEngine.ProteinPilot: ProteinPilotReader,
    SearchEngine.ProteinPilotXML: ProteinPilotXMLReader,
    SearchEngine.Mascot: MascotReader,
    SearchEngine.Comet: TPPReader,
    SearchEngine.XTandem: TPPReader,
    SearchEngine.TPP: TPPReader,
    SearchEngine.MSGFPlus: MSGFPlusReader,
    SearchEngine.Percolator: PercolatorReader,
    SearchEngine.PercolatorText: PercolatorTextReader,
}


def get_reader(search_engine: SearchEngine, ptmdb: PTMDB):
    """
    Constructs the appropriate `Reader` based on the `search_engine`.

    Args:
        search_engine: The SearchEngine used for database search.
        ptmdb: The PTMDB for `Reader` construction.

    Returns:
        The `Reader` associated with the configured search_engine.

    Raises:
        NotImplementedError.

    """
    try:
        return ENGINE_READER_MAP[search_engine](ptmdb)
    except KeyError:
        raise NotImplementedError(
            f"Cannot read search results for engine: {search_engine}"
        )


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
            if subseqs:
                yield title, ''.join(subseqs)
            title = line.rstrip()
            subseqs = []
        else:
            subseqs.append(line.rstrip())
    if subseqs:
        yield title, ''.join(subseqs)
