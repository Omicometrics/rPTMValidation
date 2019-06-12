#! /usr/bin/env python3
"""
This module provides a class for parsing TPP pepXML files.

"""
import dataclasses
import operator
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import lxml.etree as etree
from pepfrag import ModSite

from .base_reader import Reader
from .parser_exception import ParserException
from .ptmdb import PTMDB
from .search_result import PeptideType, SearchResult


@dataclasses.dataclass
class TPPSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("scores",)

    scores: Dict[str, float]


class TPPReader(Reader):  # pylint: disable=too-few-public-methods
    """
    Class to read a TPP pepXML file.

    """
    _mass_mod_names = {
        305: "iTRAQ8plex",
        58: "Cabamidomethyl",
        44: "Carbamyl"
    }

    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the reader.

        Args:
            ptmdb (PTMDB): The UniMod PTM database.

        """
        super().__init__(ptmdb)

        self.namespace = "http://regis-web.systemsbiology.net/pepXML"
        self.ns_map = {'x': self.namespace}

    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the specified pepXML file.

        Args:
            filename (str): The path to the pepXML file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:

        Raises:
            ParserException

        """
        res: List[TPPSearchResult] = []
        for event, element in etree.iterparse(filename, events=['end']):
            if (event == "end" and
                    element.tag == f"{{{self.namespace}}}spectrum_query"):
                raw_id = element.get("spectrum").split(".")
                raw_file, scan_no = raw_id[0], int(raw_id[1])
                charge = int(element.get("assumed_charge"))
                spec_id = element.get("spectrumNativeID")

                hits = [(
                    int(hit.get("hit_rank")),
                    hit.get("peptide"),
                    tuple(self._process_mods(
                        hit.find(f"{{{self.namespace}}}modification_info"))),
                    charge,
                    {s.get("name"): float(s.get("value"))
                     for s in hit.xpath("x:search_score",
                                        namespaces=self.ns_map)},
                    PeptideType.decoy
                    if hit.get("protein").startswith("DECOY_")
                    else PeptideType.normal
                )
                    for hit in element.xpath(
                        "x:search_result/x:search_hit",
                        namespaces=self.ns_map)]

                # _build_search_result has been split out as a separate method
                # such that it may be overridden for specific TPPReader
                # subclasses, e.g. Comet or X! Tandem, in future
                res.extend([
                    self._build_search_result(raw_file, scan_no, spec_id, hit)
                    for hit in hits])

        return res if not predicate else [r for r in res if predicate(r)]

    def _process_mods(self, mod_info) -> List[ModSite]:
        """
        Processes the modification elements of the pepXML search result.

        Args:
            mod_info

        Raises:
            ParserException

        """
        if mod_info is None:
            return []

        mod_nterm_mass = mod_info.get("mod_nterm_mass", None)
        nterm_mod: Optional[ModSite] = None
        if mod_nterm_mass is not None:
            mod_nterm_mass = float(mod_nterm_mass)
            name = TPPReader._mass_mod_names.get(int(mod_nterm_mass), None)

            if name is None:
                raise ParserException(
                    "Unrecognized n-terminal modification with mass "
                    f"{mod_nterm_mass}")

            nterm_mod = ModSite(mod_nterm_mass, "N-term", name)

        mods: List[ModSite] = []
        for mod in mod_info.findall(f"{{{self.namespace}}}mod_aminoacid_mass"):
            mass = (float(mod.get("static")) if "static" in mod.attrib
                    else float(mod.get("variable")))

            mod_name = self.ptmdb.get_name(mass)

            if mod_name is None:
                raise ParserException(
                    f"Unrecognized modification with mass {mass}")

            mods.append(ModSite(mass, int(mod.get("position")), mod_name))

        mods = sorted(mods, key=operator.itemgetter(1))
        if nterm_mod is not None:
            mods.insert(0, nterm_mod)

        return mods

    def _build_search_result(
            self,
            raw_file: str,
            scan_no: int,
            comb_id: str,
            hit: Tuple[int, str, Tuple[ModSite, ...], int, Dict[str, float],
                       PeptideType]) -> TPPSearchResult:
        """
        Converts a search result to a standard SearchResult.

        Returns:
            TPPSearchResult.

        """
        data_id, spec_id = comb_id.split(":")
        return TPPSearchResult(
            seq=hit[1],
            mods=list(hit[2]),
            charge=hit[3],
            spectrum=spec_id,
            dataset=data_id,
            rank=hit[0],
            pep_type=hit[5],
            theor_mz=None,
            scores=hit[4]
        )
