#! /usr/bin/env python3
"""
This module provides functions for reading ProteinPilot results
(PeptideSummary/XML) files.

"""
import csv
import re
from typing import Any, Dict, List, Optional, Tuple

from .base_reader import Reader
from . import modifications
from .search_result import PeptideType, SearchResult


MGF_TITLE_REGEX = re.compile(r"TITLE=Locus:([\d\.]+) ")


class ProteinPilotReader(Reader):
    """
    A class to read ProteinPilot PeptideSummary files.

    """
    def read(self, filename: str, min_conf: Optional[float] = None,
             **kwargs)\
            -> List[SearchResult]:
        """
        Reads the given ProteinPilot Peptide Summary file to extract useful
        information on sequence, modifications, m/z etc.

        Args:
            filename (str): The path to the Peptide Summary file.
            min_conf (float, optional): The minimum confidence threshold.

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
                if (min_conf is None or (result.confidence is not None
                                         and result.confidence >= min_conf)):
                    results.append(result)
            return results

    def _build_search_result(self, row: Dict[str, Any])\
            -> Optional[SearchResult]:
        """
        Processes the given row of a Peptide Summary file to produce a
        SearchResult entry.

        Args:
            row (dict): A row dictionary from the Peptide Summary file.

        Returns:
            A SearchResult to represent the row, or None if the modifications
            could not be identified.

        """
        mods = modifications.preparse_mod_string(row["Modifications"])

        try:
            parsed_mods = modifications.parse_mods(mods, self.ptmdb)
        except modifications.UnknownModificationException:
            return None

        return SearchResult(row["Sequence"],
                            parsed_mods,
                            int(row["Theor z"]),
                            row["Spectrum"],
                            # Rank
                            1,
                            row["Time"],
                            float(row["Conf"]),
                            float(row["Theor m/z"]),
                            float(row["Prec m/z"]),
                            PeptideType.decoy if "REVERSED" in row["Names"]
                            else PeptideType.normal)


# TODO: use XML parser
def read_proteinpilot_xml(filename: str) -> List[
        Tuple[str, List[Tuple[str, str, int, float, float, float, str]]]]:
    """
    Reads the full ProteinPilot search results in XML format.
    Note that reading this file using an XML parser does not appear to be
    straightforward due to errors related to NCNames.

    Args:
        filename (str): The path to the result XML file.

    Returns:

    """
    res: List[
        Tuple[str, List[Tuple[str, str, int, float, float, float, str]]]] = []
    with open(filename, 'r') as f:
        hits: List[Tuple[str, str, int, float, float, float, str]] = []
        t = False
        for line in f:
            sx = re.findall('"([^"]*)"', line.rstrip())
            if line.startswith('<SPECTRUM'):
                queryid = sx[6]
                pms = float(sx[4])
                t = True
            elif t:
                if line.startswith('<MATCH'):
                    sline = line.rstrip().split('=')
                    for ii, prop in enumerate(sx):
                        if sline[ii].endswith('charge'):
                            ck = int(prop)  # charge
                        if sline[ii].endswith('confidence'):
                            conf = float(prop)  # confidence
                        if sline[ii].endswith('seq'):
                            pk = prop  # sequence
                        if sline[ii].endswith('type'):
                            nk = 'decoy' if int(prop) == 1 else 'normal'
                        if sline[ii].endswith('score'):
                            sk = float(prop)
                    modj = []
                elif line.startswith('<MOD_FEATURE'):
                    j = int(sx[1])
                    modj.append('%s(%s)@%d' % (sx[0], pk[j-1], j))
                elif line.startswith('<TERM_MOD_FEATURE'):
                    if not sx[0].startswith('No'):
                        modj.insert(0, 'iTRAQ8plex@N-term')
                elif line.startswith('</MATCH>'):
                    hits.append((pk, ';'.join(modj), ck, pms, conf, sk, nk))
                elif line.startswith('</SPECTRUM>'):
                    res.append((queryid, hits))
                    hits, t = [], False
    return res
