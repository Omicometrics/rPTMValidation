#! /usr/bin/env python3
"""
A script providing utility functions for peptide modification validation.

"""
import collections
import enum
import re

from constants import AA_SYMBOLS, ELEMENT_MASSES


ModSite = collections.namedtuple("ModSite", ["mass", "site", "mod"])


class MassType(enum.Enum):
    mono = enum.auto()
    avg = enum.auto()
    
    
class UnknownModificationException(Exception):
    pass
    
    
MOD_FORMULA_REGEX = re.compile(r"(\w+)\(([0-9]+)\)")
    
    
def parse_mod_formula(formula, mass_type):
    """
    Parses the given modification chemical formula to determine the
    associated mass change.
    
    Args:
        formula (str): The modification chemical formula.
        mass_type (MassType): The mass type to calculate.
        
    Returns:
        The mass of the modification as a float.

    """
    return sum([getattr(ELEMENT_MASSES[e], mass_type.name) * int(c)
                for e, c in MOD_FORMULA_REGEX.findall(formula)])

    
def _parse_mod_string(mod_str, ptmdb, mass_type):
    """
    Parses the modification string and maps to the UniMod PTM database
    in order to extract the modification name, mass and site.
    
    Args:
        mod_str (str): The modification string derived from a ProteinPilot
                       Peptide Summary.
        ptmdb (dict): The UniMod PTM DB as a dictionary of lists.
        mass_type (MassType): The mass type to consider.
        
    Returns:
        The extracted information as a ModSite namedtuple.
        
    Raises:
        UnknownModificationException
    
    """
    mass_key = ("Monoisotopic mass" if mass_type is MassType.mono
                else "Average mass")

    mod_list = mod_str.strip().split('@')

    # Get the name of the modification
    name = mod_list[0]
    if '(' in name:
        # For cases such as Delta:H(4)C(2)(H), extract up to the final bracket
        # pair as the modification name
        name = name[:name.rfind('(')]
    
    # Get the mass change associated with the modification
    mass = None
    if name in ptmdb["PSI-MS Name"]:
        mass = ptmdb[mass_key][ptmdb["PSI-MS Name"].index(name)]
    elif name in ptmdb["Interim name"]:
        mass = ptmdb[mass_key][ptmdb["Interim name"].index(name)]
    else:
        name = name.replace(' ', '')
        if name.lower().startswith("delta"):
            mass = parse_mod_formula(name, mass_type)
        else:
            for idx, desc in enumerate(ptmdb["Description"]):
                if desc.replace(' ', '').lower() == name.lower():
                    mass = ptmdb[mass_key][idx]
                    break
                    
    if mass is None:
        raise UnknownModificationException(
            f"Failed to detect mass for modification {name}")
            
    # Get the site of the modification
    try:
        site = int(mod_list[1])
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

    
def parse_mods(mods_str, ptmdb, mass_type=MassType.mono):
    """
    Parses the modification string and maps to the UniMod DB to extract
    the modification mass, site in the sequence and name.
    
    Args:
        mods_str (str): The modification string from the ProteinPilot Peptide
                        Summary.
        ptmdb (dict): The UniMod PTM DB, from read_unimod_ptms.
        mass_type (MassType, optional): The mass type to consider, defaults
                                        to monoisotopic mass.
                                        
    Returns:
        List of modifications.
    
    Raises:
        UnknownModificationException
    
    """
    if not mods_str:
        raise UnknownModificationException("No modification string provided")

    mods = []
    if '@' in mods_str:
        mods = [_parse_mod_string(mod, ptmdb, mass_type)
                for mod in mods_str.split(';')]
    else:
        raise NotImplementedError(f"parse_mods called with {mods_str}")
        
    return mods