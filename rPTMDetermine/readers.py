#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
import collections
import csv
import email
import functools
import operator
import re
from typing import (Any, Dict, Iterable, Iterator, List, Optional,
                    TextIO, Tuple, Union)

import lxml.etree as etree

from .constants import AA_SYMBOLS, ELEMENT_MASSES, MassType


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


# TODO: use XML parser
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


class MascotReader():
    """
    Class to read Mascot dat/MIME files.

    """
    def __init__(self):
        """
        """
        pass

    def read(self, filename: str):
        """
        """
        with open(filename) as fh:
            file_content = email.parser.Parser().parse(fh)

        payloads: Dict[str, str] = {
            dict(sub._headers)["Content-Type"].split("=")[1].strip("\""):
            sub._payload for sub in file_content._payload}
            
        error_tol_search, decoy_search =\
            self.parse_parameters(payloads["parameters"])
            
    def parse_parameters(self, param_payload: str) -> Tuple[bool, bool]:
        """
        """
        params = dict(re.findall(r'(\w+)=([^\n]*)', param_payload))
        return params["ERRORTOLERANT"] == "1", params["DECOY"] == "1"
        


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
