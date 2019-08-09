#! /usr/bin/env python3
"""
This module provides functions for reading ProteinPilot results
(PeptideSummary/XML) files.

"""
import csv
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .base_reader import Reader
from . import modifications
from .search_result import PeptideType, SearchResult


MGF_TITLE_REGEX = re.compile(r"TITLE=Locus:([\d\.]+) ")


@dataclasses.dataclass(eq=True, frozen=True)
class ProteinPilotSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("time", "confidence", "prec_mz",)

    time: str
    confidence: float
    prec_mz: float


class ProteinPilotReader(Reader):  # pylint: disable=too-few-public-methods
    """
    A class to read ProteinPilot PeptideSummary files.

    """
    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the given ProteinPilot Peptide Summary file to extract useful
        information on sequence, modifications, m/z etc.

        Args:
            filename (str): The path to the Peptide Summary file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:
            The read information as a list of SearchResults.

        """
        with open(filename, newline='') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            results = []
            for row in reader:
                result = self._build_search_result(row)
                if result is None:
                    continue
                if predicate is None or predicate(result):
                    results.append(result)
            return results

    def _build_search_result(self, row: Dict[str, Any])\
            -> Optional[ProteinPilotSearchResult]:
        """
        Processes the given row of a Peptide Summary file to produce a
        SearchResult entry.

        Args:
            row (dict): A row dictionary from the Peptide Summary file.

        Returns:
            A ProteinPilotSearchResult to represent the row, or None if the
            modifications could not be identified.

        """
        mods = modifications.preparse_mod_string(row["Modifications"])

        try:
            parsed_mods = modifications.parse_mods(mods, self.ptmdb)
        except modifications.UnknownModificationException:
            return None

        return ProteinPilotSearchResult(
            seq=row["Sequence"],
            mods=tuple(parsed_mods),
            charge=int(row["Theor z"]),
            spectrum=row["Spectrum"],
            dataset=None,
            rank=1,
            pep_type=PeptideType.decoy if "REVERSED" in row["Names"]
            else PeptideType.normal,
            theor_mz=float(row["Theor m/z"]),
            time=row["Time"],
            confidence=float(row["Conf"]),
            prec_mz=float(row["Prec m/z"]))


# TODO: use XML parser
def read_proteinpilot_xml(filename: str) -> Dict[str, Dict[str, dict]]:
    """
    Reads the full ProteinPilot search results in XML format.
    Note that reading this file using an XML parser does not appear to be
    straightforward due to errors related to NCNames.

    Args:
        filename (str): The path to the result XML file.

    Returns:

    """
    res: Dict[str, Dict[str, Any]] = {}
    with open(filename, 'r') as f:
        read_identification = False
        for line in f:
            if line[0] != '<':
                continue
            rline = line.rstrip()
            if line.startswith('<SPECTRUM'):
                read_identification = True
                sx = dict(re.findall(r' (.+?)\="([^"]*)"', rline))
                queryid = sx['xml:id']
                pmz = float(sx['precursormass'])
                rank: int = 0
                hits: Dict[int, Dict[str, Any]] = {}
            elif read_identification:
                sx = dict(re.findall(r' (.+?)\="([^"]*)"', rline))
                if rline.startswith('<MATCH'):
                    rank += 1
                    ck = int(sx['charge'])  # charge
                    conf = float(sx['confidence'])   # confidence
                    seq = sx['seq']   # sequence
                    nk = 'decoy' if int(sx['type']) == 1 else 'normal'
                    sk = float(sx['score'])
                    modj: List[Tuple[Optional[float],
                                     Union[str, int], str]] = []
                elif rline.startswith('<MOD_FEATURE'):
                    modj.append((None, int(sx['pos']), sx['mod']))
                elif rline.startswith('<TERM_MOD_FEATURE'):
                    if sx['mod'][:3] != 'No ':
                        modj.insert(0, (None, 'nterm', sx['mod']))
                elif rline.startswith('</MATCH>'):
                    hits[rank] = {}
                    hits[rank]['seq'] = seq
                    hits[rank]['charge'] = ck
                    hits[rank]['modifications'] = tuple(modj)
                    hits[rank]['confidence'] = conf
                    hits[rank]['score'] = sk
                    hits[rank]['note'] = nk
                elif rline.startswith('</SPECTRUM>'):
                    if rank > 0:
                        res[queryid] = {}
                        res[queryid]['hits'] = hits
                        res[queryid]['pmz'] = pmz
                        hits = {}
                    # close reading current spectrum identifications
                    read_identification = False
    return res
