#! /usr/bin/env python3
"""
This module provides a class for reading Mascot dat (MIME) files.

"""
import collections
import dataclasses
import email.parser
import functools
import math
import operator
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.parse

from pepfrag import ModSite

from .base_reader import Reader
from .parser_exception import ParserException
from .ptmdb import PTMDB
from .search_result import PeptideType, SearchResult


QUERY_NUM_REGEX = re.compile(r"[a-z]+(\d+)")


@dataclasses.dataclass
class MascotSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("ionscore", "deltamass", "proteins", "num_matches",)

    ionscore: float
    deltamass: float
    proteins: Tuple[str]
    num_matches: int


@functools.lru_cache()
def get_identity_threshold(fdr: float, num_matches: int) -> float:
    """
    Calculates the identity threshold with the given FDR and number of
    matches.

    Returns:
        The identity threshold as a float.

    """
    # Includes empirical correction of -13
    return (-10 * math.log10(fdr / num_matches)) - 13


class MascotReader(Reader):  # pylint: disable=too-few-public-methods
    """
    Class to read Mascot dat/MIME files.

    """
    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the reader instance.

        """
        super().__init__(ptmdb)

        self.mime_parser = email.parser.Parser()

    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the given Mascot dat result file.

        Args:
            filename (str): The path to the Mascot database search results
                            file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:

        """
        with open(filename) as fh:
            file_content = self.mime_parser.parse(fh)

        # Extract all payloads from the MIME content
        payloads: Dict[str, str] = {
            str(p.get_param("name")): str(p.get_payload())
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
            res[query_id]["spectrumid"] = \
                self._get_query_title(query).split(" ")[0]

        # Extract the identified peptides for each spectrum query
        self._parse_peptides(payloads["peptides"], res, var_mods, fixed_mods,
                             error_tol_search=error_tol_search)
        if decoy_search:
            self._parse_peptides(payloads["decoy_peptides"], res, var_mods,
                                 fixed_mods, pep_type="decoy",
                                 error_tol_search=error_tol_search)

        return self._build_search_results(res, predicate)

    def _build_search_results(
            self,
            res,
            predicate: Optional[Callable[[SearchResult], bool]]) \
            -> List[MascotSearchResult]:
        """
        Converts the results to a standardized list of SearchResults.

        Returns:
            List of MascotSearchResults.

        """
        results = []
        for _, query in res.items():
            if "peptides" not in query["target"]:
                continue

            spec_id = query["spectrumid"]
            results.extend(self._build_search_result(
                query["target"], spec_id, PeptideType.normal,
                predicate))
            if "decoy" in query:
                if "peptides" not in query["decoy"]:
                    continue

                results.extend(self._build_search_result(
                    query["decoy"], spec_id, PeptideType.decoy,
                    predicate))

        return results

    def _build_search_result(
            self,
            query: Dict[str, Any],
            spec_id: str,
            pep_type: PeptideType,
            predicate: Optional[Callable[[SearchResult], bool]]) \
            -> List[MascotSearchResult]:
        """
        Converts the peptide identifications to a list of SearchResults.

        Args:
            query (dict): A dictionary of query information, including
                          peptides.
            spec_id (str): The ID of the associated spectrum.
            pep_type (PeptideType): The type of the peptide, normal/decoy.

        Returns:
            List of MascotSearchResults.

        """
        data_id, spec_id = spec_id.split(":")
        results = [
            MascotSearchResult(
                seq=peptide["sequence"],
                mods=peptide["modifications"],
                charge=query["charge"],
                spectrum=spec_id,
                dataset=data_id,
                rank=int(rank),
                pep_type=pep_type,
                theor_mz=query["mz"],
                ionscore=peptide["ionscore"],
                deltamass=peptide["deltamass"],
                proteins=peptide["proteins"],
                num_matches=query["num_matches"])
            for rank, peptide in query["peptides"].items()]
        return (results if predicate is None
                else [r for r in results if predicate(r)])

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
            match = QUERY_NUM_REGEX.match(key)
            if match is None:
                continue
            query_no = match.group(1)

            if pep_type not in res[query_no]:
                res[query_no][pep_type] = {}

            if key.startswith("qexp"):
                vals = value.split(",")
                mass, charge = float(vals[0]), int(vals[1].rstrip("+"))

                res[query_no][pep_type]["mz"] = mass
                res[query_no][pep_type]["charge"] = charge
            elif key.startswith("qmatch"):
                res[query_no][pep_type]["num_matches"] = int(value)

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

            if value == "-1":
                # No peptide assigned to the spectrum
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
            for mod_site in peptide["modifications"]:
                if mod_site.mass is not None:
                    mods.append(mod_site)
                    continue

                if mod_site.site == 0:
                    # N-terminus
                    mods.append(ModSite(mass, "nterm", mod_name))
                elif mod_site.site == len(pep_seq) + 1:
                    # C-terminus
                    mods.append(ModSite(mass, "cterm", mod_name))
                elif mod_res == pep_seq[mod_site.site - 1]:
                    mods.append(ModSite(mass, mod_site.site, mod_name))

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
