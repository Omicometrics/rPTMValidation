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
from typing import List, Optional, Pattern, Tuple

from .constants import RESIDUES


DEFAULT_RULES = os.path.join(os.path.dirname(__file__), "EnzymeRules.json")


class Proteolyzer():
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
            raise KeyError(f'Undefined or unsupported enzyme type: {enzyme}!')

        self.proteolytic_regex: Optional[Pattern] = None
        self._site_terminal = None

        self._cleavage_site = enzymes[self.enzyme]['Sites']
        self._removenaa()
        self._exceptions = enzymes[self.enzyme]['Except']
        self._terminal = enzymes[self.enzyme]['Terminal']
        self._parserules()

    def _removenaa(self):
        """
        Removes residues from the cleavage sites which are not one of the
        standard 20 amino acid residues.

        Raises:
            KeyError

        """
        sites = [''.join(set(s) & RESIDUES)
                 for s in self._cleavage_site]
        if not any(sk for sk in sites):
            raise KeyError("Unsupported cleavage sites using the enzyme "
                           f"{self.enzyme}")
        self._cleavage_site = tuple(sites)

    def _parserules(self):
        """
        Parse cleavage rules, and remove the one if the cleavage
        sites not exist in the protein sequence.
        """
        enzrules, csx, excepts, termins, siteterminal = [], [], [], [], {}
        for ii, site in enumerate(self._cleavage_site):
            if not site:
                continue
            csx.append(site)
            excepts.append(self._exceptions[ii])
            termins.append(self._terminal[ii])
            # set up string split rule
            rulej = r'([%s])' % site
            if self._exceptions[ii]:
                rulej = r'(?<![%s])' % self._exceptions[ii] + rulej \
                    if self._terminal[ii] == 'N' else\
                    rulej + r'(?![%s])' % self._exceptions[ii]
            enzrules.append(rulej)
            # get the combine direction
            for rk in site:
                siteterminal[rk] = self._terminal[ii]
        self._cleavage_site = tuple(csx)
        self._exceptions = tuple(excepts)
        self._terminal = tuple(termins)
        self.proteolytic_regex = re.compile("|".join(enzrules))
        self._site_terminal = siteterminal

    def _split_sequence(self, sequence: str) -> List[str]:
        """
        Splits the provided sequence using the rules defined.

        Args:
            sequence (str): The peptide amino acid sequence.

        Returns:
            The split sequence as a list.

        """
        if self.proteolytic_regex is None:
            raise RuntimeError("Proteolytic regex not defined")
        return [s for s in self.proteolytic_regex.split(sequence) if s]

    def is_cleaved(self, sequence: str) -> bool:
        """
        Evaluates whether a peptide sequence has been proteolytically cleaved
        using the rule associated with the current instance of the
        Proteolyzer.

        Args:
            sequence (str): The peptide amino acid sequence.

        Returns:
            A boolean indicating whether the peptide follows the cleavage
            rule.

        """
        # TODO: deal with terminal of cleavage
        return any(sequence[-1] in sites for sites in self._cleavage_site)

    def count_missed_cleavages(self, sequence: str) -> int:
        """
        Counts the number of missed cleavages in the given sequence.

        Args:
            sequence (str): The peptide amino acid sequence.

        Returns:
            An integer number of missed cleavages.

        """
        if self.proteolytic_regex is None:
            raise RuntimeError("Proteolytic regex not defined")
        # Subtract one for the main sequence itself and another one for the
        # cleavage residue at the terminus
        missed = len(re.findall(self.proteolytic_regex, sequence))
        if any(sequence[-1] in cs for cs in self._cleavage_site):
            missed -= 1
        return missed

    def cleave(self, sequence: str, numbermissed: int = 1,
               lenrange: Tuple[int, int] = (7, 60)) -> Tuple[str, ...]:
        """
        Cleavage of the input sequence using the constructed enzyme
        object, with number of missed cleavage allowed.
        Arguments
        - sequence: protein sequence in ..fasta file
        - numbermissed: number of missed cleavage allowed
        """
        if self._site_terminal is None:
            raise RuntimeError("Site terminal is not defined")

        min_len, max_len = lenrange
        # Split the sequence according to the cleavage rules of the enzyme
        split_seq = self._split_sequence(sequence)
        seq_len = len(split_seq)

        if seq_len == 1:
            return (tuple(split_seq[0],)
                    if max_len >= len(split_seq[0]) >= min_len
                    and RESIDUES.issuperset(split_seq[0]) else tuple())

        # Get all peptides with zero missed cleavage
        comb_peps = []
        for ii in range(seq_len):
            nmk, j0, ci = 0, ii, split_seq[ii]
            try:
                cterm = self._site_terminal[ci]
            except KeyError:
                # the last splitted sequence
                if ii == seq_len - 1:
                    try:
                        if self._site_terminal[split_seq[ii - 1]] == 'C':
                            comb_peps.append(split_seq[ii])
                    except KeyError:
                        break
                else:
                    continue

            # set up initial peptide sequence for searching next
            # sequence if number of missed cleavage larger than 0
            if cterm == 'C':
                if ii == 0:
                    sk = ci
                else:
                    cj = split_seq[ii - 1]
                    sk = cj + ci
                    if split_seq[ii - 1] in self._site_terminal and \
                            self._site_terminal[cj] == cterm:
                        nmk += 1
            elif cterm == 'N' and ii < seq_len - 1:
                sk = ci + split_seq[ii + 1]
                j0 += 1
                if split_seq[ii + 1] in self._site_terminal and\
                        self._site_terminal[split_seq[ii + 1]] == cterm:
                    nmk += 1
            # no missed cleavage
            comb_peps.append(sk)
            # get peptides with larger number of missed cleavage
            if nmk == numbermissed:
                continue
            for jj in range(j0 + 1, seq_len):
                cj = split_seq[jj]
                if cj in self._site_terminal:
                    if self._site_terminal[cj] == 'C':
                        sk += cj
                        nmk += 1
                        if nmk == numbermissed:
                            break
                    else:
                        nmk += 1
                        if nmk > numbermissed:
                            break
                        sk += cj
                else:
                    sk += cj
            comb_peps.append(sk)
        return tuple([x for x in comb_peps if max_len >= len(x) >= min_len
                      and RESIDUES.issuperset(x)])
