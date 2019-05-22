#! /usr/bin/env python3
"""
This module provides functions for reading ProteinPilot results
(PeptideSummary/XML) files.

"""
import collections
import csv
import re
from typing import Any, Dict, List, Tuple

PPRes = collections.namedtuple("PPRes", ["seq", "mods", "theor_z", "spec",
                                         "time", "conf", "theor_mz", "prec_mz",
                                         "accs", "names"])

MGF_TITLE_REGEX = re.compile(r"TITLE=Locus:([\d\.]+) ")


def _build_ppres(row: Dict[str, Any]) -> PPRes:
    """
    Processes the given row of a Peptide Summary file to produce a PPRes
    entry.

    Args:
        row (dict): A row dictionary from the Peptide Summary file.

    Returns:
        A PPRes namedtuple to represent the row.

    """
    return PPRes(row["Sequence"], row["Modifications"], int(row["Theor z"]),
                 row["Spectrum"], row["Time"], float(row["Conf"]),
                 float(row["Theor m/z"]), float(row["Prec m/z"]),
                 row["Accessions"], row["Names"])


def read_peptide_summary(summary_file: str, condition=None) -> List[PPRes]:
    """
    Reads the given ProteinPilot Peptide Summary file to extract useful
    information on sequence, modifications, m/z etc.

    Args:
        summary_file (str): The path to the Peptide Summary file.
        condition (func, optional): A boolean-returning function which
                                    determines whether a row should be
                                    returned.

    Returns:
        The read information as a list of PPRes NamedTuples.

    """
    with open(summary_file, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        return ([_build_ppres(r) for r in reader] if condition is None
                else [_build_ppres(r) for r in reader if condition(r)])


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
