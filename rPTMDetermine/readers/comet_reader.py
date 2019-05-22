#! /usr/bin/env python3
"""
This module provides a class for parsing Comet pepXML files.

"""
import operator
from typing import Dict, List, Optional, Tuple, Union

import lxml.etree as etree

from .ptmdb import PTMDB
from .readers import ParserException


class CometReader():
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
        self.ptmdb = ptmdb
        self.namespace = ("http://regis-web.systemsbiology.net/pepXML"
                          if namespace is None else namespace)
        self.ns_map = {'x': self.namespace}

    def read(self, filename: str) -> \
        List[Tuple[str, int, str,
                   List[Tuple[int, str,
                              Tuple[Tuple[float, Union[str, int], str], ...],
                              int, Dict[str, float], str]]]]:
        """
        Reads the specified pepXML file.

        Args:
            filename (str): The path to the pepXML file.

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

        return res

    def _process_mods(self, mod_info)\
            -> List[Tuple[float, Union[str, int], str]]:
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
        nterm_mod: Optional[Tuple[float, str, str]] = None
        if mod_nterm_mass is not None:
            mod_nterm_mass = float(mod_nterm_mass)
            name = CometReader._mass_mod_names.get(int(mod_nterm_mass), None)

            if name is None:
                raise ParserException(
                    "Unrecognized n-terminal modification with mass "
                    f"{mod_nterm_mass}")

            nterm_mod = (mod_nterm_mass, "N-term", name)

        mods: List[Tuple[float, Union[str, int], str]] = []
        for mod in mod_info.findall(f"{{{self.namespace}}}mod_aminoacid_mass"):
            mass = (float(mod.get("static")) if "static" in mod.attrib
                    else float(mod.get("variable")))

            mod_name = self.ptmdb.get_name(mass)

            if mod_name is None:
                raise ParserException(
                    f"Unrecognized modification with mass {mass}")

            mods.append((mass, int(mod.get("position")), mod_name))

        mods = sorted(mods, key=operator.itemgetter(1))
        if nterm_mod is not None:
            mods.insert(0, nterm_mod)

        return mods
