#! /usr/bin/env python3
"""
A series of functions used to read different file types.

"""
import collections
import csv

from constants import AA_SYMBOLS


Modification = collections.namedtuple("Modification", ["name", "mono"])
PPRes = collections.namedtuple("PPRes", ["seq", "mods", "theor_z", "spec",
                                         "time", "conf", "theor_mz", "prec_mz",
                                         "accs", "names"])


def _strip_line(string, nchars=2):
    """
    Strips trailing whitespace, preceeding nchars and preceeding whitespace
    from the given string.

    Args:
        string (str): The string to strip.
        nchars (int, optional): The number of preceeding characters to remove.

    Returns:
        The stripped string.

    """
    return string.rstrip()[2:].lstrip()


def read_uniprot_ptms(ptm_file):
    """
    Parses the PTM list provided by the UniProt Knowledgebase
    https://www.uniprot.org/docs/ptmlist.

    Args:
        ptm_file (str): The path to the PTM list file.

    Returns:
        A dictionary mapping residues to Modifications.

    """
    ptms = collections.defaultdict(list)
    with open(ptm_file) as fh:
        # Read the file until the entries begin at a line of underscores
        line = next(fh)
        while not line.startswith("______"):
            line = next(fh)
        mod = {}
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
    

def read_unimod_ptms(ptm_file):
    """
    Parses the PTM list provided by UNIMOD http://www.unimod.org.
    
    Args:
        ptm_file (str): The path to the UNIMOD PTM list file.
        
    Returns:
        A dictionary representation of the DB, keys representing file columns.

    """
    with open(ptm_file, newline='') as fh:
        rows = list(csv.reader(fh, delimiter='\t'))
    return {l[0]: l[1:] for l in (list(l) for l in zip(*rows))}
    
    
def _build_ppres(row):
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
    
    
def read_peptide_summary(summary_file, condition=None):
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