#! /usr/bin/env python3
"""
This module provides a function for reading the Uniprot database.

"""
import collections
from typing import Any, Dict, List

from ..constants import AA_SYMBOLS


Modification = collections.namedtuple("Modification", ["name", "mono"])


def _strip_line(string: str, nchars: int = 2) -> str:
    """
    Strips trailing whitespace, preceeding nchars and preceeding whitespace
    from the given string.

    Args:
        string (str): The string to strip.
        nchars (int, optional): The number of preceeding characters to remove.

    Returns:
        The stripped string.

    """
    return string.rstrip()[nchars:].lstrip()


def read_uniprot_ptms(ptm_file: str) -> Dict[str, List[Modification]]:
    """
    Parses the PTM list provided by the UniProt Knowledgebase
    https://www.uniprot.org/docs/ptmlist.

    Args:
        ptm_file (str): The path to the PTM list file.

    Returns:
        A dictionary mapping residues to Modifications.

    """
    ptms: Dict[str, List[Modification]] = collections.defaultdict(list)
    with open(ptm_file) as fh:
        # Read the file until the entries begin at a line of underscores
        line = next(fh)
        while not line.startswith("______"):
            line = next(fh)
        mod: Dict[str, Any] = {}
        for line in fh:
            if line.startswith("ID"):
                mod["name"] = _strip_line(line)
            elif line.startswith("TG"):
                if "Undefined" in line:
                    continue
                res_str = _strip_line(line)
                res_name = (res_str.split('-')[0] if '-' in res_str
                            else res_str[:-1])
                mod["res"] = [AA_SYMBOLS[r] for r in res_name.split(" or ")
                              if r in AA_SYMBOLS]
            elif line.startswith("MM"):
                mod["mass"] = float(_strip_line(line))
            elif line.startswith("//"):
                for res in mod.get("res", []):
                    ptms[res].append(Modification(mod["name"],
                                                  mod.get("mass", None)))
                mod = {}
    return ptms
