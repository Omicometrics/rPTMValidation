#! /usr/bin/env python3
"""
Process protein sequence database (in ..fasta file format) to
generate, such as decoy database for peptide identifications.
Enzymes and cleavage rules are set according to Mascot document at
http://www.matrixscience.com/help/enzyme_help.html.
Note that the 'none' type cleavage is not supported currently.

"""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

from .constants import RESIDUES


DEFAULT_RULES = os.path.join(os.path.dirname(__file__), "EnzymeRules.json")


class Proteolyzer:
    """
    A class for cleaving peptides according to the given enzyme cleavage rule.

    """
    def __init__(self, enzyme: str, enzyme_rules: str = DEFAULT_RULES):
        """
        Initialize the class instance with the specified enzyme cleavage rule.

        Args:
            enzyme (str): An enzyme rule defined in the enzyme_rules file.
            enzyme_rules (str, optional): The JSON file which defines the
                                          available cleavage rules.

        Raises:
            KeyError: Raised in the event that the enzyme is not defined.

        """
        self.enzyme = enzyme

        with open(enzyme_rules) as fh:
            enzymes = json.load(fh)

        # Check for valid enzyme type
        if self.enzyme not in enzymes:
            raise KeyError(f"Undefined or unsupported enzyme type: {enzyme}!")

        self.proteolytic_regex: Optional[Pattern] = None
        self._site_terminals: Optional[Dict[str, str]] = None
        self._cleavage_sites: Optional[str] = None

        cleavage_sites: List[str] = enzymes[enzyme]["Sites"]
        cleavage_sites = self._remove_invalid_site(cleavage_sites)
        self._create_rules(enzymes[enzyme], cleavage_sites)

    def _remove_invalid_site(self, cleavage_sites: Iterable[str]) -> List[str]:
        """
        Removes site from cleavage sites which is not one of the
        standard 20 amino acid residues.

        Args:
            cleavage_sites: The residues targeted for cleavage.

        Returns:
            Filtered list of cleavage sites.

        Raises:
            KeyError

        """
        sites = ["".join(set(s) & RESIDUES) for s in cleavage_sites]
        if not any(site for site in sites):
            raise KeyError("Unsupported cleavage sites using the enzyme "
                           f"{self.enzyme}")
        return sites

    def _create_rules(self, enzyme: Dict[str, Any],
                      cleavage_sites: Iterable[str]):
        """
        Create cleavage rules, and remove the one if cleavage
        sites not exist in protein sequence.

        """
        # cleavage exceptions
        excepts: Optional[List[str]] = enzyme.get("Except")
        # terminals (N / C)
        terminals: List[str] = enzyme["Terminal"]

        # create cleavage rules
        rules: List[str] = []
        sites: List[str] = []
        site_terminals: Dict[str, Any] = {}
        for i, (site, terminal) in enumerate(zip(cleavage_sites, terminals)):
            if not site:
                continue

            sites.append(site)
            # sequence split rules
            rule = f"([{site}])"
            # exceptions
            if excepts is not None:
                _rule = (f"(?<![{excepts[i]}{rule}])" if terminal == "N"
                         else f"{rule}(?![{excepts[i]}])")
            rules.append(rule)
            # combination direction: C/N terminal
            site_terminals.update((r, terminal) for r in site)

        self._cleavage_sites = "".join(sites)
        self._site_terminals = site_terminals
        self.proteolytic_regex = re.compile("|".join(rules))

    def _split_sequence(self, sequence: str) -> List[str]:
        """
        Splits the provided sequence using the rules defined.

        Args:
            sequence (str): The peptide amino acid sequence.

        Returns:
            The split sequence as a list.

        """
        if self.proteolytic_regex is None:
            raise ValueError("Proteolytic regex not defined")
        return [s for s in self.proteolytic_regex.split(sequence) if s]

    def is_cleaved(self, sequence: str) -> bool:
        """
        Evaluates whether a peptide sequence has been proteolytically
        cleaved using the rule associated with current instance of the
        Proteolyzer.

        Args:
            sequence (str): The peptide amino acid sequence.

        Returns:
            A boolean indicating whether the peptide follows the cleavage
            rule.

        Raises:
            ValueError

        """
        # TODO: deal with terminal of cleavage
        if self._cleavage_sites is not None:
            return sequence[-1] in self._cleavage_sites
        raise ValueError("Proteolyzer._cleavage_sites has not been initialized")

    def count_missed_cleavages(self, sequence: str) -> int:
        """
        Counts the number of missed cleavages in the given sequence.

        Args:
            sequence (str): The peptide amino acid sequence.

        Returns:
            An integer number of missed cleavages.

        Raises:
            ValueError

        """
        if self.proteolytic_regex is None:
            raise ValueError("Proteolytic regex is not defined")
        if self._cleavage_sites is None:
            raise ValueError("Cleavage sites are not defined")

        # Subtract one for the main sequence itself and another one
        # for the cleavage residue at the terminus
        missed = len(re.findall(self.proteolytic_regex, sequence))
        if sequence[-1] in self._cleavage_sites:
            missed -= 1
        return missed

    def cleave(self, sequence: str,
               num_missed: int = 1,
               len_range: Tuple[int, int] = (7, 30)) -> Tuple[str, ...]:
        """
        Cleavage of the input sequence using the constructed enzyme
        object, with number of missed cleavage allowed.

        Arguments:
        sequence : str
            sequence string, e.g., protein sequence in ..fasta file
         num_missed : int
            number of missed cleavages

        Raises:
            ValueError

        """
        if self._site_terminals is None:
            raise ValueError("Cleavage terminal is not defined.")
        if self._cleavage_sites is None:
            raise ValueError("Cleavage sites are not defined")

        min_len, max_len = len_range

        # Split the sequence according to the cleavage rules
        # defined by specified enzyme
        split_seq: List[str] = self._split_sequence(sequence)

        # Get all peptides
        peps: List[str] = []
        # the indices of cleavage sites
        site_index = [i for i, seq in enumerate(split_seq)
                      if seq in self._cleavage_sites]

        term: Optional[str] = None
        term_next: Optional[str] = None
        nmc: int = 0
        s: List[str] = []
        # get peptides
        for i in site_index:
            nmc, s = 0, [split_seq[i]]
            # cleavage terminal (at C/N side)
            term = self._site_terminals.get(s[0])

            # if the cleavage happens at C-terminal of the site,
            # then previous split sequence which is not a cleavage
            # site should be added
            if i > 0 and i - 1 not in site_index:
                if term == "C":
                    s.insert(0, split_seq[i - 1])
                    peps.append("".join(s))
                elif i == 1:
                    peps.append(split_seq[i - 1])

            # more missed cleavages
            for s_next in split_seq[i + 1:]:
                term_next = self._site_terminals.get(s_next)
                s.append(s_next)
                # combine subsequences
                if term_next == "C" or (term_next != "C" and term == "N"):
                    peps.append("".join(s[:-1])
                                if term_next == "N" else "".join(s))

                # count one more missed cleavage
                if (term == "C" and term_next is None) or term_next == "N":
                    nmc += 1
                    # stop search if the number of missed cleavages
                    # exceeds that allowed
                    if nmc > num_missed:
                        break

                if term_next is not None:
                    term = term_next

        if term_next is None and term == "C":
            if nmc <= num_missed:
                peps.append("".join(s))
            # the last subsequence
            peps.append(split_seq[len(split_seq) - 1])

        return tuple(set(x for x in peps if max_len >= len(x) >= min_len
                         and RESIDUES.issuperset(x)))
