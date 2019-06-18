#! /usr/bin/env python3
"""
This module provides a class for reading the UniMod database.

"""
import collections
import functools
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import lxml.etree as etree

from pepfrag import MassType
from rPTMDetermine.constants import ELEMENT_MASSES

MOD_FORMULA_REGEX = re.compile(r"(\w+)\(([0-9]+)\)")

UNIMOD_FORMULA_REGEX = re.compile(r"(\w+)\(?([0-9-]+)?\)?")


UnimodEntry = collections.namedtuple(
    "UnimodEntry",
    ["id", "name", "full_name", "mono_mass", "avg_mass", "composition"])


class PTMDB:
    """
    A class representing the UniMod PTM database.

    """
    def __init__(self, ptm_file: str):
        """
        Initializes the class object with the database connection and cursor.

        Args:
            ptm_file (str): The path to the Unimod XML file.

        """
        self.ptm_file = ptm_file

        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()

        self._initialize()

    def __del__(self):
        """
        Implements the deletion for the class object, commiting any
        outstanding changes and closing the connection to the database.

        """
        self.conn.commit()
        self.conn.close()

    def _initialize(self):
        """
        Constructs the Unimod database using the XML file.

        """
        self.cursor.execute('''CREATE TABLE mods
                               (id integer PRIMARY KEY, name text,
                                full_name text, mono_mass real, avg_mass real,
                                composition text)''')
        self.cursor.execute("CREATE INDEX name_index ON mods(name)")
        self.cursor.execute(
            "CREATE INDEX full_name_index ON mods(full_name)")

        namespace = "http://www.unimod.org/xmlns/schema/unimod_2"
        ns_map = {"x": namespace}
        entries: List[UnimodEntry] = []
        for event, element in etree.iterparse(self.ptm_file, events=["end"]):
            if event == "end" and element.tag == f"{{{namespace}}}mod":
                delta = element.xpath("x:delta", namespaces=ns_map)[0]
                entries.append(
                    UnimodEntry(
                        element.get("record_id"),
                        element.get("title"),
                        element.get("full_name"),
                        float(delta.get("mono_mass")),
                        float(delta.get("avge_mass")),
                        delta.get("composition")))

        self.cursor.executemany("INSERT INTO mods VALUES (?, ?, ?, ?, ?, ?)",
                                entries)

        self.conn.commit()

    @functools.lru_cache()
    def get_mass(self, name: str, mass_type: MassType = MassType.mono) \
            -> Optional[float]:
        """
        Retrieves the mass of the specified modification.

        Args:
            name (str): The name of the modification.
            mass_type (MassType, optional): The type of mass to retrieve.

        Returns:
            The mass as a float or None.

        """
        self.cursor.execute(
            f"SELECT {self._get_mass_col(mass_type)} FROM mods WHERE name=?",
            (name,))
        res = self.cursor.fetchone()
        return res[0] if res is not None else None

    @functools.lru_cache()
    def get_by_id(self, ptm_id: int, mass_type: MassType = MassType.mono) \
            -> Optional[Tuple[str, float]]:
        """
        Retrieves a modification entry by ID.

        Args:
            ptm_id (int): The Unimod PTM ID number.
            mass_type (MassType, optional): The type of mass to retrieve.

        Returns:
            A tuple of modification name and mass.

        """
        self.cursor.execute(
            f"SELECT name, {self._get_mass_col(mass_type)} FROM mods "
            "WHERE id=?", (ptm_id,))
        return self.cursor.fetchone()

    @functools.lru_cache()
    def get_formula(self, name: str) -> Optional[Dict[str, int]]:
        """
        Retrieves the modification formula, in terms of its elemental
        composition.

        Args:
            name (str): The name of the modification.

        Returns:
            A dictionary of element (isotope) to the number of occurrences.

        """
        self.cursor.execute("SELECT composition FROM mods WHERE name=?",
                            (name,))
        res = self.cursor.fetchone()
        if res is None:
            return None

        # Parse the composition string
        return {k: int(v) if v else 1
                for k, v in re.findall(UNIMOD_FORMULA_REGEX, res[0])}

    @functools.lru_cache()
    def get_name(self, mass: float, mass_type: MassType = MassType.mono,
                 tol: float = 0.001) -> Optional[str]:
        """
        Retrieves the name of the modification, given its mass.

        Args:
            mass (float): The modification mass.
            mass_type (MassType, optional): The mass type.
            tol (float, optional): The mass tolerance for searching.

        Returns:
            The name of the modification as a string, or None.

        """
        col = self._get_mass_col(mass_type)
        self.cursor.execute(
            f"SELECT name, {col} FROM mods WHERE {col} BETWEEN ? AND ?",
            (mass - tol, mass + tol))
        res = self.cursor.fetchone()
        return res[0] if res is not None else None

    def _get_mass_col(self, mass_type: MassType) -> str:
        """
        Returns the database column for the corresponding mass type.

        Args:
            mass_type (MassType): The type of mass (mono/avg).

        Returns:
            The column name as a string.

        """
        return "mono_mass" if mass_type is MassType.mono else "avg_mass"


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
