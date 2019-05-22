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
                    Sequence, TextIO, Tuple, Union)
import urllib.parse

import lxml.etree as etree
from pepfrag import MassType, ModSite

from .constants import AA_SYMBOLS, ELEMENT_MASSES


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
        Initialize the reader instance.

        """
        self.mime_parser = email.parser.Parser()

    def read(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Reads the given Mascot dat result file.

        Args:
            filename (str): The path to the Mascot database search results
                            file.

        Returns:

        """
        with open(filename) as fh:
            file_content = self.mime_parser.parse(fh)

        # Extract all payloads from the MIME content
        payloads: Dict[str, str] = {p.get_param("name"): p.get_payload()
                                    for p in file_content.walk()}

        # Read the relevant parameters
        error_tol_search, decoy_search =\
            self._parse_parameters(payloads["parameters"])

        # Parse the fixed and variable masses in the MIME content
        var_mods, fixed_mods = self._parse_masses(payloads["masses"])

        # Extract query information
        res: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
        self._parse_summary(payloads["summary"], res)
        if decoy_search:
            self._parse_summary(payloads["decoy_summary"], res,
                                pep_type="decoy")

        # Assign spectrum IDs to the querys
        for query_id in res.keys():
            query = payloads[f"query{query_id}"]
            res[query_id]["spectrumid"] = self._get_query_title(query)

        # Extract the identified peptides for each spectrum query
        self._parse_peptides(payloads["peptides"], res, var_mods, fixed_mods,
                             error_tol_search=error_tol_search)
        if decoy_search:
            self._parse_peptides(payloads["decoy_peptides"], res, var_mods,
                                 fixed_mods, pep_type="decoy",
                                 error_tol_search=error_tol_search)

        return res

    def _parse_parameters(self, payload: str) -> Tuple[bool, bool]:
        """
        Parses the 'parameters' payload.

        Args:
            payload (str): The content of the 'parameters' payload.

        Returns:
            Two booleans: error tolerant search and decoy search indicators.

        """
        params = self._payload_to_dict(payload)
        return params["ERRORTOLERANT"] == "1", params["DECOY"] == "1"

    def _parse_masses(self, payload: str)\
            -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Parses the 'masses' payload.

        Args:
            payload (str): The content of the 'masses' payload.

        Returns:

        """
        var_mods = {}
        fixed_mods = {}
        masses = self._payload_to_dict(payload)
        for key, mod in masses.items():
            id_match = re.match(r"[A-Za-z]+(\d+)", key)
            if id_match is None:
                continue
            mod_id = id_match.group(1)

            key_prefix = key.rstrip(mod_id)
            if key_prefix not in ["delta", "FixedMod"]:
                continue

            mass, mod_name = mod.split(",")

            mod_name, mod_res = self._get_mod_info(mod_name)

            mod_props = {
                "mass": float(mass),
                "name": mod_name,
                "residue": mod_res
            }

            if key_prefix == "delta":
                var_mods[mod_id] = mod_props
            elif key_prefix == "FixedMod":
                fixed_mods[mod_id] = mod_props
            else:
                continue

        return var_mods, fixed_mods

    def _parse_summary(self, payload: str, res, pep_type: str = "target"):
        """
        Parses a '*summary' payload.

        Args:
            payload (str): The '*summary' payload content.
            res:
            pep_type (str, optional): The result peptide type, target or decoy.

        Returns:

        """
        entries = self._payload_to_dict(payload)
        for key, value in entries.items():
            if not key.startswith("qexp"):
                continue

            vals = value.split(",")
            mass, charge = float(vals[0]), int(vals[1].rstrip("+"))

            query_no = key.lstrip("qexp")
            if not query_no:
                continue

            res[query_no][pep_type] = {
                "mz": mass,
                "charge": charge
            }

    def _parse_peptides(self, payload: str, res, var_mods, fixed_mods,
                        pep_type: str = "target",
                        error_tol_search: bool = False):
        """
        Parses a '*peptides' payload.

        Args:
            payload (str): The '*peptides' payload content.
            res:
            pep_type (str, optional): The result peptide type, target or decoy.

        Returns:

        """
        entries = self._payload_to_dict(payload)
        for key, value in entries.items():
            # Filter out unrequired keys
            if key.count("_") > 1:
                continue

            query_no, rank = self._get_query_num_rank(key)
            pep_str, protein_str = value.split(";", maxsplit=1)

            proteins = re.findall(r'"([^"]+)"', protein_str)

            seq, mods, score, delta =\
                self._parse_identification(pep_str, var_mods, fixed_mods)

            try:
                res[query_no][pep_type]["peptides"]
            except KeyError:
                res[query_no][pep_type]["peptides"] = {}

            res[query_no][pep_type]["peptides"][rank] = {
                "sequence": seq,
                "modifications": tuple(mods),
                "ionscore": score,
                "deltamass": delta,
                "proteins": tuple(proteins)
            }

        if error_tol_search:
            self._process_error_tolerant_mods(entries, res, pep_type)

    def _process_error_tolerant_mods(self, entries: Dict[str, str], res,
                                     pep_type: str):
        """
        """
        for key, value in entries.items():
            # Filter out keys unrelated to error tolerant modifications
            if not key.endswith("et_mods"):
                continue

            query_no, rank = self._get_query_num_rank(key)

            mod_info = value.split(",")
            mass = float(mod_info[0])
            mod_name, mod_res = self._get_mod_info(mod_info[1])

            peptide = res[query_no][pep_type]["peptides"][rank]
            pep_seq = peptide["sequence"]

            mods = []
            for ms in peptide["modifications"]:
                if ms.mass is not None:
                    mods.append(ms)
                    continue

                if ms.site == 0:
                    # N-terminus
                    mods.append(ModSite(mass, "nterm", mod_name))
                elif ms.site == len(pep_seq) + 1:
                    # C-terminus
                    mods.append(ModSite(mass, "cterm", mod_name))
                elif mod_res == pep_seq[ms.site - 1]:
                    mods.append(ModSite(mass, ms.site, mod_name))

            peptide["modifications"] = tuple(mods)

    def _parse_identification(self, info_str: str, var_mods, fixed_mods)\
            -> Tuple[str, List[ModSite], float, float]:
        """
        Parses a peptide identification string to extract sequence,
        modifications and score.

        Args:

        Returns:

        """
        split_str = info_str.split(",")
        # Peptide sequence and modification string
        pep_str, mods_str = [split_str[i] for i in [4, 6]]
        # delta is the difference between the theoretical and experimentally
        # determined peptide masses; score is the IonScore
        delta, score = [float(split_str[i]) for i in [2, 7]]

        # Process modifications
        mods, term_mods, mod_nterm, mod_cterm = [], [], False, False
        for idx, char in enumerate(mods_str):
            if char == "0":
                # No modification applied at this position
                continue

            # X indicates that this is the modification site for error
            # tolerant search
            if char == "X":
                mods.append(ModSite(None, idx, None))
                if idx == 0:
                    mod_nterm = True
                elif idx == len(mods_str) - 1:
                    mod_cterm = True
                continue

            # Standard modifications
            mod_mass, mod_name = var_mods[char]["mass"], var_mods[char]["name"]
            if idx == 0:
                term_mods.append(ModSite(mod_mass, "nterm", mod_name))
            elif idx == len(mods_str) - 1:
                term_mods.append(ModSite(mod_mass, "cterm", mod_name))
            else:
                mods.append(ModSite(mod_mass, idx, mod_name))

        # Apply fixed modifications
        for mod_info in fixed_mods.values():
            mod_mass, mod_name = mod_info["mass"], mod_info["name"]

            if mod_info["residue"] == "N-term":
                if not mod_nterm:
                    term_mods.append(ModSite(mod_mass, "nterm", mod_name))
            elif mod_info["residue"] == "C-term":
                if not mod_cterm:
                    term_mods.append(ModSite(mod_mass, "cterm", mod_name))
            else:
                mods.extend([ModSite(mod_mass, idx + 1, mod_name)
                             for idx, char in enumerate(pep_str)
                             if char == mod_info["residue"]])

        mods = term_mods + sorted(mods, key=operator.itemgetter(1))

        return pep_str, mods, score, delta

    def _get_mod_info(self, mod_str: str) -> Sequence[str]:
        """
        Extracts the modification name and residue from a string.

        Args:
            mod_str (str): A string in the format 'NAME (RES)'

        Returns:
            The modification name and the target residue.

        Raises:
            ParserException.

        """
        match = re.match(r"(\w+) \(([\w\-_]+)\).*", mod_str)
        if match is None:
            raise ParserException(f"Invalid modification: {mod_str}")
        return match.groups()

    def _get_query_title(self, payload):
        """
        Parses a 'query*' payload to extract the title.

        Args:
            payload (str): A 'query*' payload's content.

        Returns:
            The title spectrum ID as a string.

        """
        entries = self._payload_to_dict(payload)
        return urllib.parse.unquote(entries["title"])

    def _get_query_num_rank(self, string: str) -> Sequence[str]:
        """
        Extracts the query number and peptide rank from a payload key.

        Args:
            string (str): The payload key, in the form qN_pM*.

        Returns:
            The query number and rank as strings.

        Raises:
            ParserException.

        """
        match = re.search(r"q(\d+)_p(\d+)", string)
        if match is None:
            raise ParserException(
                f"Invalid string passed to _get_query_num_rank: {string}")
        return match.groups()

    def _payload_to_dict(self, payload: str) -> Dict[str, str]:
        """
        Converts a payload, newline-separated string into a dictionary.

        Args:
            payload (str): The payload content.

        Returns:
            A dictionary mapping the LHS of the = to the RHS.

        """
        return dict(re.findall(r'(\w+)=([^\n]*)', payload))


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
