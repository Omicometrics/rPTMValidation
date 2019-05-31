#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
from typing import Iterable, List, TextIO, Tuple

from rPTMDetermine.base_config import SearchEngine

from . import comet_reader
from . import mascot_reader
from . import protein_pilot_reader
from .ptmdb import PTMDB


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
        return protein_pilot_reader.ProteinPilotReader(ptmdb)
    if search_engine is SearchEngine.Comet:
        return comet_reader.CometReader(ptmdb)
    if search_engine is SearchEngine.Mascot:
        return mascot_reader.MascotReader(ptmdb)
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