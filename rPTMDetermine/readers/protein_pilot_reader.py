#! /usr/bin/env python3
"""
This module provides functions for reading ProteinPilot results
(PeptideSummary/XML) files.

"""
import collections
import csv
import dataclasses
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import lxml.etree as etree

from pepfrag import ModSite

from . import (
    modifications,
    utilities
)
from .base_reader import Reader
from .search_result import PeptideType, SearchResult
from .ptmdb import PTMDB, ModificationNotFoundException


ITRAQ_COL_REGEX = re.compile(r"(%Err )?\d{3}:\d{3}")
ITRAQ_PEAK_COL_REGEX = re.compile(r'(?:Err|Area) \d{3}')


@dataclasses.dataclass(eq=True, frozen=True)
class _ProteinPilotSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = (
        "time",
        "confidence",
        "prec_mz",
        "proteins",
        "accessions",
        "itraq_peaks",
    )

    time: str
    confidence: float
    prec_mz: float
    accessions: Optional[Tuple[str, ...]]
    itraq_peaks: Optional[Dict[int, Tuple[float, float]]]


@dataclasses.dataclass(eq=True, frozen=True)
class ProteinPilotSearchResult(_ProteinPilotSearchResult):

    __slots__ = (
        "proteins",
        "itraq_ratios",
        "background",
        "used_in_quantitation",
    )

    proteins: Optional[str]
    itraq_ratios: Optional[Dict[str, float]]
    background: Optional[float]
    used_in_quantitation: Optional[bool]


@dataclasses.dataclass(eq=True, frozen=True)
class ProteinPilotXMLSearchResult(_ProteinPilotSearchResult):

    __slots__ = (
        "byscore",
        "eval",
        "mod_prob",
    )

    byscore: Optional[float]
    eval: Optional[float]
    mod_prob: Optional[float]


class ProteinPilotReader(Reader):  # pylint: disable=too-few-public-methods
    """
    A class to read ProteinPilot PeptideSummary files.

    """
    def read(
            self,
            filename: str,
            predicate: Optional[Callable[[SearchResult], bool]] = None,
            read_itraq_ratios: bool = False,
            read_itraq_peaks: bool = False,
            **kwargs
    ) -> List[ProteinPilotSearchResult]:
        """
        Reads the given ProteinPilot Peptide Summary file to extract useful
        information on sequence, modifications, m/z etc.

        Args:
            filename: The path to the Peptide Summary file.
            predicate: An optional predicate to filter results.
            read_itraq_ratios: Whether to include the iTRAQ quantitation ratios
                               in the returned results.
            read_itraq_peaks: Whether to include the iTRAQ peak areas in the
                              returned results. Note that these are returned as
                              a tuple of (peak area, peak area error).

        Returns:
            The read information as a list of ProteinPilotSearchResults.

        """
        with open(filename, newline='') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            results = []
            itraq_cols = []
            itraq_peak_cols = []
            for row in reader:
                if read_itraq_ratios and not itraq_cols:
                    itraq_cols = self._get_itraq_cols(row)
                if read_itraq_peaks and not itraq_peak_cols:
                    itraq_peak_cols = self._get_itraq_peak_cols(row)
                result = self._build_search_result(
                    row, itraq_cols, itraq_peak_cols
                )
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

    @staticmethod
    def _get_itraq_peak_cols(row: Dict[str, Any]) -> List[str]:
        """
        Extracts the names of the iTRAQ peak area columns from a row of the
        PeptideSummary file.

        """
        return [k for k in row.keys() if ITRAQ_PEAK_COL_REGEX.match(k)]

    def _build_search_result(
            self,
            row: Dict[str, Any],
            itraq_cols: List[str],
            itraq_peak_cols: List[str]
    ) -> Optional[ProteinPilotSearchResult]:
        """
        Processes the given row of a Peptide Summary file to produce a
        ProteinPilotSearchResult entry.

        Args:
            row (dict): A row dictionary from the Peptide Summary file.

        Returns:
            A ProteinPilotSearchResult to represent the row, or None if the
            modifications could not be identified.

        """
        mods = modifications.preparse_mod_string(row["Modifications"])

        try:
            parsed_mods = modifications.parse_mods(mods, self.ptmdb)
        except modifications.UnknownModificationException as ex:
            logging.warning(ex)
            return None

        try:
            used = bool(int(row['Used']))
        except ValueError:
            used = None

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
            itraq_peaks=(
                {k: float(row[k]) for k in itraq_peak_cols if row[k]}
                if itraq_peak_cols else None
            ),
            proteins=row["Names"],
            accessions=row["Accessions"],
            itraq_ratios=(
                {k: float(row[k]) for k in itraq_cols if row[k]}
                if itraq_cols else None
            ),
            background=(
                float(row["Background"]) if 'Background' in row else None
            ),
            used_in_quantitation=used
        )


TempResult = collections.namedtuple(
    "TempResult",
    ("spec_id", "prec_mz", "rank", "seq", "charge", "mods", "conf",
     "score", "eval", "mod_prob", "pep_type", "rt", "itraq_peaks"))


@dataclasses.dataclass(eq=True, frozen=True)
class _ProteinPilotXMLTempResult(SearchResult):
    __slots__ = (
        "confidence",
        "prec_mz",
        "byscore",
        "eval",
        "mod_prob",
        "time",
        "itraq_peaks",
    )

    confidence: float
    prec_mz: float
    byscore: Optional[float]
    eval: Optional[float]
    mod_prob: Optional[float]
    time: str
    itraq_peaks: Optional[Dict[int, Tuple[float, float]]]


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
             **kwargs) -> List[ProteinPilotXMLSearchResult]:
        """
        Reads the given ProteinPilot XML file to extract useful
        information on sequence, modifications, m/z etc.

        Args:
            filename: The path to the XML file.
            predicate: An optional predicate to filter results.
            read_itraq_peaks: Whether to extract the iTRAQ peak information from
                              the XML. Note that these are returned as a tuple
                              of (peak area, peak area error).

        Returns:
            The read information as a list of ProteinPilotXMLSearchResults.

        """
        res: Dict[str, _ProteinPilotXMLTempResult] = {}
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
                    rank += 1
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

                    temp_result = _ProteinPilotXMLTempResult(
                        seq=match_element.get("seq"),
                        mods=tuple(mods),
                        charge=int(match_element.get("charge")),
                        spectrum=spec_id,
                        dataset=None,
                        rank=rank,
                        pep_type=pep_type,
                        theor_mz=None,
                        confidence=float(match_element.get("confidence")),
                        prec_mz=prec_mz,
                        byscore=float(match_element.get("score")),
                        eval=float(match_element.get("eval")),
                        mod_prob=float(match_element.get("mod_prob")),
                        time=str(retention_time),
                        itraq_peaks=itraq_peaks
                    )

                    if predicate is None or predicate(temp_result):
                        res[match_id] = temp_result

                element.clear()
                for ancestor in element.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]

            elif element.tag == "SEQUENCE":
                match_ids = []
                for pep_element in element.findall("PEPTIDE"):
                    match_ids.extend(pep_element.get("matches").split(","))
                protein_ids = [e.get("protein")
                               for e in element.findall("PROTEIN_CONTEXT")]
                for match_id in match_ids:
                    match_protein_map[match_id] = protein_ids

                element.clear()
                for ancestor in element.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]

        for match_id, r in res.items():
            yield ProteinPilotXMLSearchResult(
                    seq=r.seq,
                    mods=r.mods,
                    charge=r.charge,
                    spectrum=r.spectrum,
                    dataset=None,
                    rank=r.rank,
                    theor_mz=None,
                    pep_type=r.pep_type,
                    time=r.time,
                    confidence=r.confidence,
                    prec_mz=r.prec_mz,
                    accessions=tuple(match_protein_map.get(match_id, [])),
                    byscore=r.byscore,
                    eval=r.eval,
                    mod_prob=r.mod_prob,
                    itraq_peaks=r.itraq_peaks
            )

    @staticmethod
    def read_biases(filename: str) -> Dict[Tuple[str, str], float]:
        """
        Reads the given ProteinPilot XML file to extract the iTRAQ ratio
        bias coefficients.

        Args:
            filename (str): The path to the ProteinPilot XML file.

        """
        biases: Dict[Tuple[str, str], float] = {}
        found_bias = False
        # Since the biases are found towards the end of the XML file and parsing
        # the file to find them can be expensive, we reverse iterate over the
        # lines of the file instead, stopping when there are no further BIAS
        # elements
        for line in utilities.reverse_readline(filename):
            if '<BIAS' in line:
                found_bias = True
                attrs = utilities.xml_line_to_dict(line)
                biases[(attrs['nominator'], attrs['denominator'])] =\
                    float(attrs['coefficient'])
            elif found_bias:
                # We have finished processing the BIAS elements
                break

        return biases

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
