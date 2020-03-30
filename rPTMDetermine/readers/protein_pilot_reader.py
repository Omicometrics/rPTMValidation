#! /usr/bin/env python3
"""
This module provides functions for reading ProteinPilot results
(PeptideSummary/XML) files.

"""
import collections
import csv
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import lxml.etree as etree

from pepfrag import ModSite

from .base_reader import Reader
from . import modifications
from .search_result import PeptideType, SearchResult
from .ptmdb import PTMDB, ModificationNotFoundException


ITRAQ_COL_REGEX = re.compile(r"(%Err )?\d{3}:\d{3}")


@dataclasses.dataclass(eq=True, frozen=True)
class _ProteinPilotSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("time", "confidence", "prec_mz", "proteins", "accessions",
                 "byscore", "eval", "mod_prob",)

    time: str
    confidence: float
    prec_mz: float
    proteins: Optional[str]
    accessions: Optional[Tuple[str, ...]]
    byscore: Optional[float]
    eval: Optional[float]
    mod_prob: Optional[float]


@dataclasses.dataclass(eq=True, frozen=True)
class ProteinPilotSearchResult(_ProteinPilotSearchResult):

    __slots__ = ("itraq_ratios",)

    itraq_ratios: Optional[Dict[str, float]]


@dataclasses.dataclass(eq=True, frozen=True)
class ProteinPilotXMLSearchResult(_ProteinPilotSearchResult):

    __slots__ = ("itraq_peaks",)

    itraq_peaks: Optional[Dict[int, Tuple[float, float]]]


class ProteinPilotReader(Reader):  # pylint: disable=too-few-public-methods
    """
    A class to read ProteinPilot PeptideSummary files.

    """
    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             read_itraq_ratios: bool = False,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the given ProteinPilot Peptide Summary file to extract useful
        information on sequence, modifications, m/z etc.

        Args:
            filename (str): The path to the Peptide Summary file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.
            read_itraq_ratios (bool, optional): Whether to include the iTRAQ
                quantitation ratios in the returned SearchResults.

        Returns:
            The read information as a list of SearchResults.

        """
        with open(filename, newline='') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            results = []
            itraq_cols = []
            for row in reader:
                if read_itraq_ratios and not itraq_cols:
                    itraq_cols = self._get_itraq_cols(row)
                result = self._build_search_result(row, itraq_cols)
                if result is None:
                    continue
                if predicate is None or predicate(result):
                    results.append(result)
            return results

    @staticmethod
    def _get_itraq_cols(row: Dict[str, Any]) -> List[str]:
        """
        Extracts the names of the iTRAQ ratio columns from a row of the
        PeptideSummary file.

        """
        return [k for k in row.keys() if ITRAQ_COL_REGEX.match(k)]

    def _build_search_result(self, row: Dict[str, Any],
                             itraq_cols: List[str])\
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
            prec_mz=float(row["Prec m/z"]),
            proteins=row["Names"],
            accessions=row["Accessions"],
            byscore=None,
            eval=None,
            mod_prob=None,
            itraq_ratios=({k: float(row[k]) for k in itraq_cols if row[k]}
                          if itraq_cols else None)
        )


TempResult = collections.namedtuple(
    "TempResult",
    ("spec_id", "prec_mz", "rank", "seq", "charge", "mods", "conf",
     "score", "eval", "mod_prob", "pep_type", "rt", "itraq_peaks"))


class ProteinPilotXMLReader(Reader):  # pylint: disable=too-few-public-methods
    """
    A class to read ProteinPilot XML files.

    """

    id_namespace = "http://www.w3.org/XML/1998/namespace"

    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the reader.

        Args:
            ptmdb (PTMDB): The UniMod PTM database.

        """
        super().__init__(ptmdb)

    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             read_itraq_peaks: bool = False,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the given ProteinPilot XML file to extract useful
        information on sequence, modifications, m/z etc.

        Args:
            filename (str): The path to the XML file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:
            The read information as a list of SearchResults.

        """
        res: Dict[str, TempResult] = {}
        match_protein_map: Dict[str, List[str]] = {}
        context = etree.iterparse(filename, events=["end"], recover=True,
                                  encoding="iso-8859-1")
        for event, element in context:
            if element.tag == "SPECTRUM":
                # Remove the last number from the ProteinPilot spectrum ID
                spec_id = ".".join(
                    element.get(f"{{{self.id_namespace}}}id").split(".")[:-1])
                prec_mz = float(element.get("precursormass"))
                retention_time = float(element.get("elution"))

                itraq_peaks: Optional[Dict[int, Tuple[float, float]]] =\
                    self._get_itraq_peaks(element) if read_itraq_peaks else None

                rank = 0
                for match_element in element.findall("MATCH"):
                    rank = rank + 1
                    pep_type = (PeptideType.normal
                                if int(match_element.get("type")) == 0
                                else PeptideType.decoy)
                    match_id = match_element.get(f"{{{self.id_namespace}}}id")

                    try:
                        mods: List[ModSite] = list(
                            self._parse_mods(match_element, "MOD_FEATURE"))
                        mods.extend(
                            self._parse_mods(match_element, "TERM_MOD_FEATURE",
                                             "nterm"))
                    except ModificationNotFoundException:
                        continue

                    res[match_id] = TempResult(
                        spec_id, prec_mz, rank,
                        match_element.get("seq"),
                        int(match_element.get("charge")),
                        mods,
                        float(match_element.get("confidence")),
                        float(match_element.get("score")),
                        float(match_element.get("eval")),
                        float(match_element.get("mod_prob")),
                        pep_type,
                        retention_time,
                        itraq_peaks)

                element.clear()

            elif element.tag == "SEQUENCE":
                match_ids = []
                for pep_element in element.findall("PEPTIDE"):
                    match_ids.extend(pep_element.get("matches").split(","))
                protein_ids = [e.get("protein")
                               for e in element.findall("PROTEIN_CONTEXT")]
                for match_id in match_ids:
                    match_protein_map[match_id] = protein_ids

                element.clear()

        return [ProteinPilotXMLSearchResult(
                    seq=r.seq,
                    mods=tuple(r.mods),
                    charge=r.charge,
                    spectrum=r.spec_id,
                    dataset=None,
                    rank=r.rank,
                    theor_mz=None,
                    pep_type=r.pep_type,
                    time=r.rt,
                    confidence=r.conf,
                    prec_mz=r.prec_mz,
                    proteins=None,
                    accessions=tuple(match_protein_map.get(match_id, [])),
                    byscore=r.score,
                    eval=r.eval,
                    mod_prob=r.mod_prob,
                    itraq_peaks=r.itraq_peaks)
                for match_id, r in res.items()]

    @staticmethod
    def _get_itraq_peaks(element) -> Dict[int, Tuple[float, float]]:
        """

        """
        peaks: Dict[int, Tuple[float, float]] = {}
        peaks_element = element.find("ITRAQPEAKS")
        if peaks_element is not None:
            for line in peaks_element.text.split("\n"):
                if any(s for s in line):
                    tag, peak_area, err_peak_area = line.split("\t")
                    peaks[int(float(tag))] = \
                        (float(peak_area), float(err_peak_area))
        return peaks

    def _parse_mods(self, element, mod_xml_tag: str,
                    fixed_site: Optional[str] = None):
        """
        Raises:
            ModificationNotFoundException

        """
        for mod in element.findall(mod_xml_tag):
            name = mod.get("mod")
            if not name.startswith("No "):
                yield ModSite(
                    self.ptmdb.get_mass(name),
                    int(mod.get("pos")) if fixed_site is None else fixed_site,
                    name)
