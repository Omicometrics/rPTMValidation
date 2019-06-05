#! /usr/bin/env python3
"""
This module provides a class for parsing Comet pepXML files.

"""
import dataclasses
import operator
from typing import Callable, List, Optional, Sequence

import lxml.etree as etree
from pepfrag import ModSite

from .base_reader import Reader
from .parser_exception import ParserException
from .ptmdb import PTMDB
from .search_result import PeptideType, SearchResult


@dataclasses.dataclass
class CometSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ()


class CometReader(Reader):  # pylint: disable=too-few-public-methods
    """
    Class to read a Comet pepXML file.

    """
    _mass_mod_names = {
        305: "iTRAQ8plex",
        58: "Cabamidomethyl",
        44: "Carbamyl"
    }

    def __init__(self, ptmdb: PTMDB,
                 namespace: Optional[str] = None):
        """
        Initialize the reader.

        Args:
            ptmdb (PTMDB): The UniMod PTM database.
            namespace (str, optional): The pepXML namespace.

        """
        super().__init__(ptmdb)

        self.namespace = ("http://regis-web.systemsbiology.net/pepXML"
                          if namespace is None else namespace)
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
        res = []
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
                    "decoy" if hit.get("protein").startswith("DECOY_")
                    else "normal"
                    )
                        for hit in element.xpath(
                            "x:search_result/x:search_hit",
                            namespaces=self.ns_map)]

                if hits:
                    res.append((raw_file, scan_no, spec_id, hits))

        return self._build_search_results(res)

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
            name = CometReader._mass_mod_names.get(int(mod_nterm_mass), None)

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

    def _build_search_results(self, res) -> List[CometSearchResult]:
        """
        Converts the search results to the standard SearchResult class.

        Returns:
            List of CometSearchResults.

        """
        results = []
        for id_set in res:
            data_id, spec_id = id_set[2].split(":")
            results.extend([
                CometSearchResult(
                    seq=hit[1],
                    mods=list(hit[2]),
                    charge=hit[3],
                    spectrum=spec_id,
                    dataset=data_id,
                    rank=hit[0],
                    pep_type=PeptideType[hit[5]],
                    theor_mz=None) for hit in id_set[3]])
        return results
