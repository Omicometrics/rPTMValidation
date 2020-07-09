#! /usr/bin/env python3
"""
This module provides a class for parsing TPP pepXML files.

"""
import dataclasses
import operator
from typing import Any, Callable, Dict, List, Optional

import lxml.etree as etree
from pepfrag import AA_MASSES, ModSite

from .base_reader import Reader
from .parser_exception import ParserException
from .ptmdb import ModificationNotFoundException, PTMDB
from .search_result import PeptideType, SearchResult


@dataclasses.dataclass(eq=True, frozen=True)
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
             **kwargs) -> List[TPPSearchResult]:
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
                spec_id = self._get_id(element)

                hits = [
                    self._extract_hit(hit, charge)
                    for hit in element.xpath(
                        'x:search_result/x:search_hit',
                        namespaces=self.ns_map
                    )
                ]

                # _build_search_result has been split out as a separate method
                # such that it may be overridden for specific TPPReader
                # subclasses, e.g. Comet or X! Tandem, in future
                res.extend([
                    self._build_search_result(raw_file, scan_no, spec_id, hit)
                    for hit in hits
                ])

        return res if not predicate else [r for r in res if predicate(r)]

    def _get_id(self, query_element) -> str:
        """
        Extracts the spectrum ID from the query XML element.

        """
        spec_id = query_element.get("spectrumNativeID")
        if spec_id is None:
            return f"0.1.{query_element.get('start_scan')}"
        return spec_id.split(":")[1]

    def _extract_hit(self, hit_element, charge: int) -> Dict[str, Any]:
        """
        Parses the search_hit XML element to extract relevant information.

        """
        return {
            'rank': int(hit_element.get("hit_rank")),
            'seq': hit_element.get("peptide"),
            'mods': tuple(self._process_mods(
                hit_element.get('peptide'),
                hit_element.find(f"{{{self.namespace}}}modification_info")
            )),
            'charge': charge,
            'scores': {
                s.get("name"): float(s.get("value"))
                for s in hit_element.xpath(
                    "x:search_score", namespaces=self.ns_map
                )
            },
            'pep_type': PeptideType.decoy
            if hit_element.get("protein").startswith("DECOY_")
            else PeptideType.normal
        }

    def _process_mods(self, seq: str, mod_info) -> List[ModSite]:
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
                try:
                    name = self.ptmdb.get_name(mod_nterm_mass)
                except ModificationNotFoundException:
                    name = 'unknown'

            nterm_mod = ModSite(mod_nterm_mass, "N-term", name)

        mods: List[ModSite] = []
        for mod in mod_info.findall(f"{{{self.namespace}}}mod_aminoacid_mass"):
            for attrib in ['mass', 'static', 'variable']:
                if attrib in mod.attrib:
                    mass = float(mod.get(attrib))
                    break
            else:
                raise ParserException(
                    'Unable to determine modification mass from '
                    f'mod_amino_acid_mass with attributes: {mod.attrib}'
                )

            site = int(mod.get('position'))
            mod_mass = mass - AA_MASSES[seq[site - 1]].mono
            try:
                mod_name = self.ptmdb.get_name(mod_mass)
            except ModificationNotFoundException:
                mod_name = 'unknown'

            mods.append(ModSite(mod_mass, site, mod_name))

        mods = sorted(mods, key=operator.attrgetter('site'))
        if nterm_mod is not None:
            mods.insert(0, nterm_mod)

        return mods

    @staticmethod
    def _build_search_result(
            raw_file: str,
            scan_no: int,
            spec_id: str,
            hit: Dict[str, Any]
    ) -> TPPSearchResult:
        """
        Converts a search result to a standard SearchResult.

        Returns:
            TPPSearchResult.

        """
        return TPPSearchResult(
            seq=hit['seq'],
            mods=hit['mods'],
            charge=hit['charge'],
            spectrum=spec_id,
            dataset=None,
            rank=hit['rank'],
            pep_type=hit['pep_type'],
            theor_mz=None,
            scores=hit['scores']
        )
