#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
import collections
import csv
import re
from typing import Any, Dict, Iterable, List, TextIO, Tuple

from constants import AA_SYMBOLS, ELEMENT_MASSES, MassType


Modification = collections.namedtuple("Modification", ["name", "mono"])
PPRes = collections.namedtuple("PPRes", ["seq", "mods", "theor_z", "spec",
                                         "time", "conf", "theor_mz", "prec_mz",
                                         "accs", "names"])

MGF_TITLE_REGEX = re.compile(r"TITLE=Locus:([\d\.]+) ")

UNIMOD_FORMULA_REGEX = re.compile(r"(\w+)\(?([0-9-]+)?\)?")

MOD_FORMULA_REGEX = re.compile(r"(\w+)\(([0-9]+)\)")


class ParserException(Exception):
    """
    A custom exception to be raised during file parse errors.

    """


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


class PTMDB():
    """
    A class representing the UniMod PTM DB data structure.

    """
    _mass_keys = ['Monoisotopic mass', 'Average mass']
    _name_keys = ['PSI-MS Name', 'Interim name']
    _desc_key = 'Description'
    _comp_key = 'Composition'

    def __init__(self, ptm_file):
        """
        Initializes the class by setting up the composed dictionary.

        Args:
            ptm_file (str): The path to the UniMod PTM file.

        """
        self._data = {
            'Monoisotopic mass': [],
            'Average mass': [],
            PTMDB._comp_key: [],
            # Each of the below keys store a dictionary mapping their
            # position in the above lists
            'PSI-MS Name': {},
            'Interim name': {},
            PTMDB._desc_key: {}
        }

        with open(ptm_file, newline='') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            for row in reader:
                self.add_entry(row)

    def add_entry(self, entry):
        """
        Adds a new entry to the database.

        Args:
            entry (dict): A row from the UniMod PTB file.

        """
        pos = len(self._data[PTMDB._mass_keys[0]])
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


def _build_ppres(row: Dict[str, Any]) -> PPRes:
    """
    Processes the given row of a Peptide Summary file to produce a PPRes
    entry.

    Args:
        row (dict): A row dictionary from the Peptide Summary file.

    Returns:
        A PPRes namedtuple to represent the row.

    """
    return PPRes(row["Sequence"], row["Modifications"], int(row["Theor z"]),
                 row["Spectrum"], row["Time"], float(row["Conf"]),
                 float(row["Theor m/z"]), float(row["Prec m/z"]),
                 row["Accessions"], row["Names"])


def read_peptide_summary(summary_file: str, condition=None) -> List[PPRes]:
    """
    Reads the given ProteinPilot Peptide Summary file to extract useful
    information on sequence, modifications, m/z etc.

    Args:
        summary_file (str): The path to the Peptide Summary file.
        condition (func, optional): A boolean-returning function which
                                    determines whether a row should be
                                    returned.

    Returns:
        The read information as a list of PPRes NamedTuples.

    """
    with open(summary_file, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        return ([_build_ppres(r) for r in reader] if condition is None
                else [_build_ppres(r) for r in reader if condition(r)])


def read_proteinpilot_xml(filename: str) -> List[
        Tuple[str, List[Tuple[str, str, int, float, float, float, str]]]]:
    """
    Reads the full ProteinPilot search results in XML format.
    Note that reading this file using an XML parser does not appear to be
    straightforward due to errors related to NCNames.

    Args:
        filename (str): The path to the result XML file.

    Returns:

    """
    res: List[
        Tuple[str, List[Tuple[str, str, int, float, float, float, str]]]] = []
    with open(filename, 'r') as f:
        hits: List[Tuple[str, str, int, float, float, float, str]] = []
        t = False
        for line in f:
            sx = re.findall('"([^"]*)"', line.rstrip())
            if line.startswith('<SPECTRUM'):
                queryid = sx[6]
                pms = float(sx[4])
                t = True
            elif t:
                if line.startswith('<MATCH'):
                    sline = line.rstrip().split('=')
                    for ii, prop in enumerate(sx):
                        if sline[ii].endswith('charge'):
                            ck = int(prop)  # charge
                        if sline[ii].endswith('confidence'):
                            conf = float(prop)  # confidence
                        if sline[ii].endswith('seq'):
                            pk = prop  # sequence
                        if sline[ii].endswith('type'):
                            nk = 'decoy' if int(prop) == 1 else 'normal'
                        if sline[ii].endswith('score'):
                            sk = float(prop)
                    modj = []
                elif line.startswith('<MOD_FEATURE'):
                    j = int(sx[1])
                    modj.append('%s(%s)@%d' % (sx[0], pk[j-1], j))
                elif line.startswith('<TERM_MOD_FEATURE'):
                    if not sx[0].startswith('No'):
                        modj.insert(0, 'iTRAQ8plex@N-term')
                elif line.startswith('</MATCH>'):
                    hits.append((pk, ';'.join(modj), ck, pms, conf, sk, nk))
                elif line.startswith('</SPECTRUM>'):
                    res.append((queryid, hits))
                    hits, t = [], False
    return res


def read_fasta_sequences(fasta_file: TextIO) -> Iterable[Tuple[str, str]]:
    """
    Retrieves sequences from the input fasta_file.

    Args:
        fasta_file (TextIOWrapper): An open file handle to the fasta file.

    Yields:
        Sequences from the input file.

    """
    subseqs: List[str] = []
    for line in fasta_file:
        if line.startswith('>'):
            title = line.rstrip()
            if subseqs:
                yield title, ''.join(subseqs)
            subseqs = []
        else:
            subseqs.append(line.rstrip())
    if subseqs:
        yield title, ''.join(subseqs)
