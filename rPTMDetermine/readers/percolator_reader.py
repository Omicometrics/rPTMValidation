#! /usr/bin/env python3
"""
This module provides a class for reading MS-GF+ mzIdentML files.

"""
import dataclasses
import re
import csv
import os
from typing import Callable, List, Optional, Sequence, Tuple

import lxml.etree as etree
from overrides import overrides

from pepfrag import ModSite

from .base_reader import Reader
from .parser_exception import ParserException
from .ptmdb import PTMDB
from .search_result import PeptideType, SearchResult


@dataclasses.dataclass(eq=True, frozen=True)
class PercolatorSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("prec_mz", "svm_score", "q_value", "pep", "p_value")

    prec_mz: float
    svm_score: float
    q_value: float
    pep: float
    p_value: float


class PercolatorReader(Reader):  # pylint: disable=too-few-public-methods
    """
    Class to read percolator results files.

    """

    mod_regex = re.compile(r"\[([^\]]*)\]")

    def __init__(self, ptmdb: PTMDB,
                 namespace: str = "http://per-colator.com/percolator_out/15"):
        """
        Initialize the reader instance.

        """
        super().__init__(ptmdb)

        self.namespace = namespace
        self.ns_map = {'x': self.namespace}

    @overrides
    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> List[PercolatorSearchResult]:
        """
        Reads the given mzIdentML result file.

        Args:
            filename (str): The path to the percolator search results file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:

        """
        res: List[PercolatorSearchResult] = []
        context = etree.iterparse(
            filename, events=("end",), tag=self._fix_tag("psm"))
        for event, element in context:
            psm_id = element.get(self._fix_tag("psm_id"))
            pep_type = (PeptideType.normal
                        if element.get(self._fix_tag("decoy")) == "false"
                        else PeptideType.decoy)
            seq, mods = self._process_seq(
                element.find(self._fix_tag("peptide_seq")).get("seq"))
            scan, charge, rank = self._parse_id(psm_id)
            res.append(
                PercolatorSearchResult(
                    seq=seq,
                    mods=tuple(mods),
                    charge=charge,
                    spectrum=f"0.1.{scan}",
                    dataset=None,
                    rank=rank,
                    pep_type=pep_type,
                    theor_mz=float(self._get_tag(element, "calc_mass")),
                    prec_mz=float(self._get_tag(element, "exp_mass")),
                    svm_score=float(self._get_tag(element, "svm_score")),
                    q_value=float(self._get_tag(element, "q_value")),
                    pep=float(self._get_tag(element, "pep")),
                    p_value=float(self._get_tag(element, "p_value"))
                )
            )

        return res

    def _process_seq(self, peptide_seq: str) -> Tuple[str, List[ModSite]]:
        """
        Processes the sequence, including modification information, to extract
        the amino acid sequence and the ModSites.

        Args:
            peptide_seq (str): The peptide sequence with inline modifications.

        Returns:
            The clean peptide sequence and a list of ModSites.

        """
        mods: List[ModSite] = []
        while "[" in peptide_seq:
            match = self.mod_regex.search(peptide_seq)
            if match is None:
                raise ParserException(
                    f"Failed to parse peptide sequence {peptide_seq}")

            mod_id = int(match.group(1).split(":")[1])

            mod = self.ptmdb.get_by_id(mod_id)
            if mod is None:
                raise ParserException(
                    f"No modification found with Unimod ID {mod_id}")

            idx = peptide_seq.index("[")
            mods.append(ModSite(mod[1], "N-term" if idx == 0 else idx, mod[0]))

            peptide_seq = self.mod_regex.sub("", peptide_seq, count=1)
        return peptide_seq, mods

    def _parse_id(self, psm_id: str) -> Tuple[str, int, int]:
        """
        Parses the psm_id attribute to extract scan number, rank and charge
        state information.

        Args:
            psm_id (str): The psm_id attribute text.

        Returns:
            A tuple of (scan number, charge, rank).

        """
        # From the Percolator source code
        # https://github.com/percolator/percolator/blob/master/src/converters
        # /MsgfplusReader.cpp
        # Format is <FILEPATH>_SII_<MSGF ID>_<SCAN#>_<CHARGE>_<RANK>
        content = psm_id.split("_")[-3:]
        return (content[0], int(content[1]), int(content[2]))

    def _get_tag(self, element, tag: str) -> str:
        """
        Retrieves the text content of the given tag for the given XML
        element.

        Args:
            element (XML element)
            tag (str): The tag to retrieve.

        Returns:
            Text content.

        """
        return element.find(self._fix_tag(tag)).text

    def _fix_tag(self, tag: str) -> str:
        """
        Prepends the namespace to the given XML tag.

        Args:
            tag (str): An XML tag.

        Returns:
            The XML tag, prepended by the XML namespace.

        """
        return f"{{{self.namespace}}}{tag}"


class PercolatorTextReader(Reader):  # pylint: disable=too-few-public-methods
    """
    Class to read percolator results files.

    """

    mod_regex = re.compile(r"\[([^\]]*)\]")

    @overrides
    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the given mzIdentML result file.

        Args:
            filename (str): The path to the percolator search results file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:

        """

        file_index = self._get_file_from_index(filename)

        res: List[PercolatorSearchResult] = []
        with open(filename, "r") as f:
            frd = csv.DictReader(f, delimiter="\t")
            for r in frd:
                pep_type = PeptideType.normal
                seq, mods = self._process_seq(r["sequence"])
                res.append(
                    PercolatorSearchResult(
                        seq=seq,
                        mods=tuple(mods),
                        charge=int(r["charge"]),
                        spectrum=f"0.1.{r['scan']}",
                        dataset=file_index[r["file_idx"]],
                        rank=1,
                        pep_type=pep_type,
                        theor_mz=float(r["peptide mass"]),
                        prec_mz=float(r["spectrum precursor m/z"]),
                        svm_score=float(r["percolator score"]),
                        q_value=float(r["percolator q-value"]),
                        pep=float(r["percolator PEP"]),
                        p_value=None
                    )
                )

        return res
        
    def _process_seq(self, peptide_seq: str) -> Tuple[str, List[ModSite]]:
        """
        Processes the sequence, including modification information, to extract
        the amino acid sequence and the ModSites.

        Args:
            peptide_seq (str): The peptide sequence with inline modifications.

        Returns:
            The clean peptide sequence and a list of ModSites.

        """
        mods: List[ModSite] = []
        while "[" in peptide_seq:
            match = self.mod_regex.search(peptide_seq)
            if match is None:
                raise ParserException(
                    f"Failed to parse peptide sequence {peptide_seq}")

            mod_mass = float(match.group(1))

            mod = self.ptmdb.get_name(mod_mass, tol=0.01)
            if mod is None:
                raise ParserException(
                    f"No modification found with Unimod ID {mod_mass}")

            idx = peptide_seq.index("[")
            mod_mass_accurate = self.ptmdb.get_mass(mod)
            mods.append(ModSite(mod_mass_accurate,
                                "N-term" if idx == 0 else idx,
                                mod))

            peptide_seq = self.mod_regex.sub("", peptide_seq, count=1)
        return peptide_seq, mods
        
    def _get_file_from_index(self, filename):
        """
        Get file name from the log file to translate the index
        to file name.
        """
        res_files = {}
        file_dir = os.path.dirname(filename)
        log_file = os.path.join(file_dir, "percolator.log.txt")

        if not os.path.isfile(log_file):
            raise IOError("Log file not found.")

        with open(log_file, "r") as f:
            for line in f:
                if "Assigning index" in line:
                    idx, full_path = re.findall("INFO: Assigning index (.*) to (.*).",
                                                line.rstrip())[0]
                    res_files[idx] = os.path.basename(full_path)

        return res_files