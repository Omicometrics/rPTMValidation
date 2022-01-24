#! /usr/bin/env python3
"""
This is a basic module for parsing pep.xml files.

"""
import dataclasses
import lxml.etree as etree

from html import unescape
from typing import Any, Callable, Dict, List, Optional

from pepfrag import AA_MASSES, ModSite

from .base_reader import Reader
from .parser_exception import ParserException
from .ptmdb import ModificationNotFoundException, PTMDB
from .search_result import PeptideType, SearchResult


@dataclasses.dataclass(eq=True, frozen=True)
class TPPSearchResult(SearchResult):

    __slots__ = ("scores",)

    scores: Dict[str, float]


class TPPBaseReader(Reader):
    """
    Class to read a TPP pepXML file.

    """
    mod_term_mass_correct = {"mod_cterm_mass": 17.00274,
                             "mod_nterm_mass": 1.007825}

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
            if element.tag == f"{{{self.namespace}}}spectrum_query":
                raw_id = element.get("spectrum").split(".")
                raw_file, scan_no = raw_id[0], int(raw_id[1])
                charge = int(element.get("assumed_charge"))
                spec_id = self._get_id(element)

                hits = [self._extract_hit(h, charge)
                        for h in element.xpath('x:search_result/x:search_hit',
                                               namespaces=self.ns_map)]

                # _build_search_result has been split out as a separate method
                # such that it may be overridden for specific TPPReader
                # subclasses, e.g. Comet or X! Tandem, in future
                res.extend([
                    self._build_search_result(raw_file, scan_no, spec_id, hit)
                    for hit in hits
                ])

                element.clear()
                for ancestor in element.xpath("ancestor-or-self::*"):
                    while ancestor.getprevious() is not None:
                        if ancestor.getparent() is None:
                            break
                        del ancestor.getparent()[0]

        return res if not predicate else [r for r in res if predicate(r)]

    def _get_id(self, query_element) -> str:
        """
        Extracts the spectrum ID from the query XML element.

        """
        spec_id = query_element.get("spectrumNativeID")
        if spec_id is None:
            return query_element.get('start_scan')
        return unescape(spec_id)

    def _extract_hit(self, hit_element, charge: int) -> Dict[str, Any]:
        """
        Parses the search_hit XML element to extract relevant information.

        """
        return {
            'rank': int(hit_element.get("hit_rank")),
            'seq': hit_element.get("peptide"),
            'mods': tuple(self._process_mods(
                hit_element.get("peptide"),
                hit_element.find(f"{{{self.namespace}}}modification_info")
            )),
            'charge': charge,
            'scores': {
                s.get("name"): float(s.get("value"))
                for s in hit_element.xpath(
                    "x:search_score", namespaces=self.ns_map)},
            'pep_type': PeptideType.decoy
            if hit_element.get("protein").startswith("DECOY_")
            else PeptideType.normal
        }

    def _process_mods(self, seq, mod_info) -> List[ModSite]:
        """
        Processes the modification elements of the pepXML search result.

        Args:
            seq: sequence
            mod_info: modification element

        Raises:
            ParserException

        """
        def _get_mod_name(m):
            try:
                name = self.ptmdb.get_name(m)
            except ModificationNotFoundException:
                name = 'unknown'
            return name

        mods: List[ModSite] = []

        if mod_info is None:
            return mods

        # terminal modifications
        for attrib in ["mod_nterm_mass", "mod_cterm_mass"]:
            mod_term_mass = mod_info.get(attrib, None)
            if mod_term_mass is not None:
                mod_term_mass = float(mod_term_mass)
                mod_term_mass -= self.mod_term_mass_correct[attrib]
                # get modification name
                mods.append(ModSite(
                    mod_term_mass,
                    "nterm" if attrib == "mod_nterm_mass" else "cterm",
                    _get_mod_name(mod_term_mass)))

        # amino acid modifications
        for mod in mod_info.findall(f"{{{self.namespace}}}mod_aminoacid_mass"):
            for attrib in ['static', 'variable', 'mass']:
                if attrib in mod.attrib:
                    mass = float(mod.get(attrib))
                    break
            else:
                raise ParserException(
                    'Unable to determine modification mass from '
                    f'mod_amino_acid_mass with attributes: {mod.attrib}'
                )

            site = int(mod.get('position'))
            # the mass is from 'mass' content, which is the mass of residue
            # plus modification mass, thus should be corrected
            if attrib == "mass":
                # for some results, the sequence contains a unknown residue.
                if seq[site - 1] in AA_MASSES:
                    mass -= AA_MASSES[seq[site - 1]].mono
                    mods.append(ModSite(mass, site, _get_mod_name(mass)))
                else:
                    mods.append(ModSite(None, site, "unknown"))
            else:
                mods.append(ModSite(mass, site, _get_mod_name(mass)))

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
