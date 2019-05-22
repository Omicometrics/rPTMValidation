#! /usr/bin/env python3
"""
A script providing functions for parsing and processing the modifications
applied to peptides.

"""
from typing import List, Sequence, Union

from pepfrag import MassType, ModSite

from .readers import PTMDB


class UnknownModificationException(Exception):
    """
    A custom exception to be used if an unknown modification type is found.

    """


def get_mod_mass(mod_sites: Sequence[ModSite], mod: str) -> float:
    """
    Retrieves the mass of the target modification.

    Args:
        mod_sites (list): A list of ModSite namedtuples.
        mod (str): The name of the modification for which to retrieve
                   the mass.

    Returns:
        The mass associated with the modification as a float.

    Raises:
        UnknownModificationException

    """
    for mod_site in mod_sites:
        if mod_site.mod == mod:
            return mod_site.mass
    raise UnknownModificationException(
        f"Mass not found for modification {mod} within {mod_sites}")


def preparse_mod_string(mods: str) -> str:
    """
    Pre-parses the modification string to a list of ModSites.

    Args:
        mod_str (str): The modification string.

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
    mod_list = mod_str.strip().split('@')

    # Get the name of the modification
    name = mod_list[0]
    if '(' in name:
        # For cases such as Delta:H(4)C(2)(H), extract up to the final bracket
        # pair as the modification name
        name = name[:name.rfind('(')]

    # Get the mass change associated with the modification
    mass = ptmdb.get_mass(name, mass_type)

    if mass is None:
        raise UnknownModificationException(
            f"Failed to detect mass for modification {name}")

    # Get the site of the modification
    try:
        site: Union[str, int] = int(mod_list[1])
    except ValueError:
        site_str = mod_list[1].lower()
        if site_str.startswith('c') and "term" in site_str:
            site = "cterm"
        elif site_str.startswith('n') and "term" in site_str:
            site = "nterm"
        else:
            raise UnknownModificationException(
                f"Failed to detect site for modification {mod_str}")

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

    mods: List[ModSite] = []
    if '@' in mods_str:
        # Ignore modifications that begin with "No ", since these reflect the
        # absence of a modification, e.g. quantitative label
        mods = [_parse_mod_string(mod, ptmdb, mass_type)
                for mod in mods_str.split(';') if not mod.startswith("No ")]
    elif "|" in mods_str:
        mods = [_parse_bar_mod_string(mod)
                for mod in mods_str.split(",")]
    else:
        raise NotImplementedError(f"parse_mods called with {mods_str}")

    return mods
