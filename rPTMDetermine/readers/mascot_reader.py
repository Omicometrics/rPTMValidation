#! /usr/bin/env python3
"""
This module provides a class for reading Mascot dat (MIME) files.

"""
import collections
import email
import operator
import re
from typing import Any, Dict, List, Sequence, Tuple
import urllib.parse

from pepfrag import ModSite

from .readers import ParserException


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
