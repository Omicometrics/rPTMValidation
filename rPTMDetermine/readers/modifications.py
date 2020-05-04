#! /usr/bin/env python3
"""
A script providing functions for parsing and processing the modifications
applied to peptides.

"""
import logging
from typing import List, Union

from pepfrag import MassType, ModSite

from .ptmdb import ModificationNotFoundException, PTMDB


class UnknownModificationException(Exception):
    """
    A custom exception to be used if an unknown modification type is found.

    """


def preparse_mod_string(mods: str) -> str:
    """
    Pre-parses the modification string to a list of ModSites.

    Args:
        mods (str): The modification string.

    Returns:
        list of ModSites.

    """
    mods = mods.replace(" ", "")
    mods = ';'.join(
        m.split("ProteinTerminal")[1] if m.startswith("ProteinTerminal")
        else m for m in mods.split(';') if not m.startswith("No"))
    return mods


def _parse_mod_string(mod_str: str, ptmdb: PTMDB, mass_type: MassType)\
        -> ModSite:
    """
    Parses the modification string and maps to the UniMod PTM database
    in order to extract the modification name, mass and site.

    Args:
        mod_str (str): The modification string derived from a ProteinPilot
                       Peptide Summary.
        ptmdb (PTMDB): The UniMod PTM DB as a dictionary of lists.
        mass_type (MassType): The mass type to consider.

    Returns:
        The extracted information as a ModSite namedtuple.

    Raises:
        UnknownModificationException

    """
    name, site = mod_str.strip().split('@')

    # Get the site of the modification
    try:
        site: Union[str, int] = int(site)
    except ValueError:
        site_str = site.lower()
        if site_str.startswith('c') and "term" in site_str:
            site = "cterm"
        elif site_str.startswith('n') and "term" in site_str:
            site = "nterm"
        else:
            msg = f"Failed to detect site for modification {mod_str}"
            logging.warning(msg)
            raise UnknownModificationException(msg)

    if '(' in name and site != "nterm":
        # For cases such as Delta:H(4)C(2)(H), extract up to the final bracket
        # pair as the modification name, so long as the final brackets do not
        # contain a number
        idx = name.rfind('(')
        if not name[idx + 1].isdigit():
            name = name[:idx]

    # Get the mass change associated with the modification
    try:
        mass = ptmdb.get_mass(name, mass_type)
    except ModificationNotFoundException:
        msg = f"Failed to detect mass for modification {name}"
        logging.warning(msg)
        raise UnknownModificationException(msg)

    return ModSite(mass, site, name)


def _parse_bar_mod_string(mod_str: str) -> ModSite:
    """
    Parses the vertical bar-separated modification string.

    Args:
        mod_str (str): The modification string, vertical bar-separated.

    Returns:
        The extracted information as a ModSite namedtuple.

    """
    mod_list = mod_str.strip().split('|')

    site: Union[str, int] = mod_list[1]
    try:
        site = int(site)
    except ValueError:
        pass

    return ModSite(float(mod_list[0]), site, mod_list[2])


def parse_mods(mods_str: str, ptmdb: PTMDB,
               mass_type: MassType = MassType.mono) -> List[ModSite]:
    """
    Parses the modification string and maps to the UniMod DB to extract
    the modification mass, site in the sequence and name.

    Args:
        mods_str (str): The modification string from the ProteinPilot Peptide
                        Summary.
        ptmdb (PTMDB): The UniMod PTM DB.
        mass_type (MassType, optional): The mass type to consider, defaults
                                        to monoisotopic mass.

    Returns:
        List of modifications.

    Raises:
        UnknownModificationException

    """
    if not mods_str:
        return []

    if '@' in mods_str:
        # Ignore modifications that begin with "No ", since these reflect the
        # absence of a modification, e.g. quantitative label
        mods = [_parse_mod_string(mod, ptmdb, mass_type)
                for mod in mods_str.split(';') if not mod.startswith("No ")]
    elif "|" in mods_str:
        mods = [_parse_bar_mod_string(mod)
                for mod in mods_str.split(",")]
    else:
        raise NotImplementedError(
            f"parse_mods called with incompatible string: {mods_str}")

    return mods
