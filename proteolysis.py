#! /usr/bin/env python3
"""
Process protein sequence database (in ..fasta file format) to
generate, such as decoy database for peptide identifications.
Enzymes and cleavage rules are set according to Mascot document at
http://www.matrixscience.com/help/enzyme_help.html.
Note that the 'none' type cleavage is not supported currently.

"""
import re

from constants import RESIDUES
        

# TODO: encode this in a JSON file to make it more easily customizable?
ENZYME = {'Trypsin': {'site':('KR',), 'except':('P',), 'terminal':('C',)},
    'Trypsin/P': {'site':('KR',), 'except':('',), 'terminal':('C',)},
    'Arg-C': {'site':('R',), 'except':('P',), 'terminal':('C',)},
    'Asp-N': {'site':('BD',), 'except':('',), 'terminal':('N',)},
    'Asp-N_ambic': {'site':('DE',), 'except':('',), 'terminal':('N',)},
    'Chymotrypsin': {'site':('FYWL',), 'except':('P',), 'terminal':('C',)},
    'CNBr': {'site':('M',), 'except':('',), 'terminal':('C',)},
    'CNBr+Trypsin':{'site': ('M','KR'),
        'except':('','P'), 'terminal': ('C','C')},
    'Formic_acid': {'site':('D',), 'except':('',), 'terminal':('C',)},
    'Lys-C': {'site':('K',), 'except':('P',),'terminal':('C',)},
    'Lys-C/P': {'site':('K',), 'except':('',), 'terminal':('C',)},
    'LysC+AspN':{'site':('K', 'DB'),
        'except':('P','P'), 'terminal':('C', 'N')},
    'Lys-N': {'site':('K',), 'except':('',), 'terminal':('N',)},
    'PepsinA': {'site':('FL',), 'except':('',), 'terminal':('C',)},
    'semiTrypsin': {'site':('KR',), 'except':('P',), 'terminal':('C',)},
    'TrypChymo': {'site':('FYWLKR',), 'except':('P',), 'terminal':('C',)}, 
    'TrypsinMSIPI': {'site':('KR',), 'except':('P',), 'terminal':('C',)},
    'TrypsinMSIPI/P': {'site':('KR',), 'except':('',), 'terminal':('C',)},
    'V8-DE': {'site':('BDEZ',), 'except':('P',), 'terminal':('C',)},
    'V8-E': {'site':('EZ',), 'except':('P',), 'terminal':('C',)}
}
    
class Proteolyzer():
    def __init__(self, enzyme):
        """
        Enzyme project using the proteolytic cleavage rules set up
        in proteolysis dictionary in ENZYME, according to the input
        "enzyme".
        Argument
        - enzyme: proteolysis, if not in the dictionary, raise an exception
        """
        self.enzyme = enzyme

        # Check for valid enzyme type
        if not self.enzyme in ENZYME:
            raise KeyError('Undefined or unsupported enzyme type!')

        self._cleavage_site = ENZYME[enzyme]['site']
        self._removenaa()
        self._exceptions = ENZYME[enzyme]['except']
        self._terminal = ENZYME[enzyme]['terminal']
        self._parserules()
        
    def _removenaa(self):
        """
        Removes unrecognized residues not any one of 20 common residues.
        
        Raises:
            KeyError

        """
        sites = [''.join(set(sk) & RESIDUES) for sk in self._cleavage_site]
        if not any(len(sk) > 0 for sk in sites):
            raise KeyError("Unsupported cleavage sites using the enzyme "
                           f"{self.enzyme}")
        self._cleavage_site = tuple(sites)
    
    def _parserules(self):
        """
        Parse cleavage rules, and remove the one if the cleavage
        sites not exist in the protein sequence.
        """
        enzrules, csx, excepts, termins, siteterminal = [], [], [], [], {}
        for i, sk in enumerate(self._cleavage_site):
            if not sk: continue
            csx.append(sk)
            excepts.append(self._exceptions[i])
            termins.append(self._terminal[i])
            # set up string split rule
            rulej = r'([%s])'%sk
            if self._exceptions[i]:
                rulej = r'(?<![%s])'%self._exceptions[i]+rulej\
                    if self._terminal[i]=='N' else\
                    rulej+r'(?![%s])'%self._exceptions[i]
            enzrules.append(rulej)
            # get the combine direction
            for rk in sk:
                siteterminal[rk] = self._terminal[i]
        self._cleavage_site = tuple(csx)
        self._exceptions = tuple(excepts)
        self._terminal = tuple(termins)
        self.proteolytic_regex = re.compile('|'.join(enzrules))
        self._site_terminal = siteterminal
        
    def _split_sequence(self, sequence):
        """
        Splits the provided sequence using the rules defined.
        
        Args:
            sequence (str): The peptide amino acid sequence.
            
        Returns:
            The split sequence as a list.

        """
        return [s for s in self.proteolytic_regex.split(sequence) if s]
        
    def count_missed_cleavages(self, sequence):
        """
        Counts the number of missed cleavages in the given sequence.
        
        Args:
            sequence (str): The peptide amino acid sequence.
            
        Returns:
            An integer number of missed cleavages.

        """
        return len(self._split_sequence(sequence))
            
    def cleave(self, sequence, numbermissed=1, lenrange=(7, 60)):
        """
        Cleavage of the input sequence using the constructed enzyme
        object, with number of missed cleavage allowed.
        Arguments
        - sequence: protein sequence in ..fasta file
        - numbermissed: number of missed cleavage allowed
        """
        min_len, max_len = lenrange
        # Split the sequence according to the cleavage rules of the enzyme
        split_seq = self._split_sequence(sequence)
        n = len(split_seq)

        if n == 1:
            return ((split_seq[0]) if max_len >= len(split_seq[0]) >= min_len
                    and RESIDUES.issuperset(split_seq[0]) else ())

        # Get all peptides with zero missed cleavage
        comb_peps = []
        for i in range(n):
            nmk, j0, ci = 0, i, split_seq[i]
            try:
                cterm = self._site_terminal[ci]
            except KeyError:
                # the last splitted sequence
                if i == n - 1:
                    try:
                        if self._site_terminal[split_seq[i - 1]] == 'C':
                            comb_peps.append(split_seq[i])
                    except KeyError:
                        break
                else:
                    continue

            # set up initial peptide sequence for searching next
            # sequence if number of missed cleavage larger than 0
            if cterm == 'C':
                if i == 0:
                    sk = ci
                else:
                    cj = split_seq[i - 1]
                    sk = cj + ci
                    if split_seq[i - 1] in self._site_terminal and \
                        self._site_terminal[cj] == cterm:
                        nmk += 1
            elif cterm == 'N' and i < n - 1:
                sk = ci + split_seq[i + 1]
                j0 += 1
                if split_seq[i + 1] in self._site_terminal and\
                    self._site_terminal[split_seq[i + 1]] == cterm:
                    nmk += 1
            # no missed cleavage
            comb_peps.append(sk)
            # get peptides with larger number of missed cleavage
            if nmk == numbermissed:
                continue
            for j in range(j0 + 1, n):
                cj = split_seq[j]
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