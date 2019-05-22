#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
from typing import Iterable, List, TextIO, Tuple


class ParserException(Exception):
    """
    A custom exception to be raised during file parse errors.

    """


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
