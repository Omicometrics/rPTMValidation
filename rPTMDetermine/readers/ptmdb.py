#! /usr/bin/env python3
"""
This module provides a class for reading the UniMod database.

"""
import csv
import functools
import re
from typing import Iterator, Optional, Tuple

from pepfrag import MassType
from rPTMDetermine.constants import ELEMENT_MASSES

MOD_FORMULA_REGEX = re.compile(r"(\w+)\(([0-9]+)\)")

UNIMOD_FORMULA_REGEX = re.compile(r"(\w+)\(?([0-9-]+)?\)?")


class PTMDB():
    """
    A class representing the UniMod PTM DB data structure.

    """
    _mono_mass_key = "Monoisotopic mass"
    _avg_mass_key = "Average mass"
    _mass_keys = [_mono_mass_key, _avg_mass_key]
    _psi_name_key = "PSI-MS Name"
    _interim_name_key = "Interim name"
    _name_keys = [_psi_name_key, _interim_name_key]
    _desc_key = 'Description'
    _comp_key = 'Composition'

    def __init__(self, ptm_file):
        """
        Initializes the class by setting up the composed dictionary.

        Args:
            ptm_file (str): The path to the UniMod PTM file.

        """
        self._data = {
            PTMDB._mono_mass_key: [],
            PTMDB._avg_mass_key: [],
            PTMDB._comp_key: [],
            # Each of the below keys store a dictionary mapping their
            # position in the above lists
            PTMDB._psi_name_key: {},
            PTMDB._interim_name_key: {},
            PTMDB._desc_key: {}
        }

        with open(ptm_file, newline='') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            for row in reader:
                self._add_entry(row)

        self._reversed = {key: {v: k for k, v in self._data[key].items()}
                          for key in PTMDB._name_keys}

    def __iter__(self) -> Iterator[Tuple[str, float, float]]:
        """
        Implements iteration as a generator for the PTMDB class.

        """
        for idx, mono in enumerate(self._data[PTMDB._mono_mass_key]):
            name = (self._reversed[PTMDB._psi_name_key][idx]
                    if idx in self._reversed[PTMDB._psi_name_key]
                    else self._reversed[PTMDB._interim_name_key][idx])
            yield (name, mono, self._data[PTMDB._avg_mass_key][idx])

    def _add_entry(self, entry):
        """
        Adds a new entry to the database.

        Args:
            entry (dict): A row from the UniMod PTB file.

        """
        pos = len(self._data[PTMDB._mono_mass_key])
        for key in PTMDB._mass_keys:
            self._data[key].append(float(entry[key]))
        for key in PTMDB._name_keys:
            self._data[key][entry[key]] = pos
        self._data[PTMDB._desc_key][entry[key].replace(' ', '').lower()] = pos
        self._data[PTMDB._comp_key].append(entry[PTMDB._comp_key])

    def _get_idx(self, name: str) -> int:
        """
        Retrieves the index of the specified modification, i.e. its position
        in the mass and composition lists.

        Args:
            name (str): The name of the modification.

        Returns:
            The integer index of the modification, or None.

        """
        # Try matching either of the two name fields, using PSI-MS Name first
        for key in PTMDB._name_keys:
            idx = self._data[key].get(name, None)
            if idx is not None:
                return idx

        # Try matching the description
        name = name.replace(' ', '')
        return self._data[PTMDB._desc_key].get(name.lower(), None)

    @functools.lru_cache()
    def get_mass(self, name, mass_type=MassType.mono):
        """
        Retrieves the mass of the specified modification.

        Args:
            name (str): The name of the modification.
            mass_type (MassType, optional): The type of mass to retrieve.

        Returns:
            The mass as a float or None.

        """
        mass_key = (PTMDB._mass_keys[0] if mass_type is MassType.mono
                    else PTMDB._mass_keys[1])

        idx = self._get_idx(name)
        if idx is not None:
            return self._data[mass_key][idx]

        # Try matching the modification name
        name = name.replace(' ', '')
        if name.lower().startswith("delta"):
            return parse_mod_formula(name, mass_type)

        return None

    @functools.lru_cache()
    def get_formula(self, name):
        """
        Retrieves the modification formula, in terms of its elemental
        composition.

        Args:
            name (str): The name of the modification.

        Returns:
            A dictionary of element (isotope) to the number of occurrences.

        """
        idx = self._get_idx(name)
        if idx is None:
            return None

        # Parse the composition string
        return {k: int(v) if v else 1
                for k, v in re.findall(UNIMOD_FORMULA_REGEX,
                                       self._data[PTMDB._comp_key][idx])}

    @functools.lru_cache()
    def get_name(self, mass: float, mass_type: MassType = MassType.mono)\
            -> Optional[str]:
        """
        Retrieves the name of the modification, given its mass.

        Args:
            mass (float): The modification mass.
            mass_type (MassType, optional): The mass type.

        Returns:
            The name of the modification as a string.

        """
        key = (PTMDB._mono_mass_key if mass_type is MassType.mono
               else PTMDB._avg_mass_key)
        for idx, db_mass in enumerate(self._data[key]):
            if abs(mass - db_mass) < 0.001:
                return (self._reversed[PTMDB._psi_name_key][idx]
                        if idx in self._reversed[PTMDB._psi_name_key]
                        else self._reversed[PTMDB._interim_name_key][idx])
        return None


def parse_mod_formula(formula: str, mass_type: MassType) -> float:
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
