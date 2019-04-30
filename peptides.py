#! /usr/bin/env python3
"""
A set of functions to be used for processing peptide sequences.

"""
import collections
import sys

from constants import AA_MASSES, FIXED_MASSES
import modifications

sys.path.append("../pepfrag")
from ion_generators import IonType


Ion = collections.namedtuple("Ion", ["mass", "label", "pos"])


def calculate_mz(seq, mods, charge):
    """
    Calculates the mass/charge ratio of a peptide given its sequence,
    modifications and charge state.

    Args:
        seq (str): The peptide sequence.
        mods (list of ModSites): The modifications applied to the peptide.
        charge (int): The charge state of the peptide.

    Returns:
        float: The mass/charge ratio of the peptide.

    """
    return (sum(AA_MASSES[r].mono for r in seq) +
            sum(m for m, _, _ in mods) + FIXED_MASSES["H2O"]) / charge


def merge_seq_mods(seq, mods):
    """
    Inserts modifications into the corresponding locations of the peptide
    sequence.

    Args:
        seq (str): The peptide sequence.
        mods (list): A list of ModSites indicating the peptide modifications.

    Returns:
        string: The peptide sequence with modification details.

    """
    if isinstance(mods, str):
        mod_str = modifications.preparse_mod_string(mods)
        mods = [modifications.ModSite(None, *m.split("@")[::-1])
                for m in mod_str.split(";") if m]

    if not mods:
        return seq

    # Convert nterm and cterm sites to their corresponding indices
    seqlen = len(seq)
    positions = {"nterm": 0, "n-term": 0, "N-term": 0, "cterm": seqlen, "c-term": seqlen, "C-term": seqlen}
    mod_sites = [(name, int(positions.get(site, site))) for _, site, name in mods]

    # Sort the modifications by site index
    mod_sites.sort(key=lambda x: x[1], reverse=True)
    
    # Insert the modifications into the peptide sequence
    seq_list = list(seq)
    for name, site in mod_sites:
        seq_list.insert(int(site), f"[{name}]")

    return ''.join(seq_list)


def get_by_ion_mzs(peptide):
    """
    Get the b/y-type fragment ions for the peptide.

    """
    return [ion.mass for ion in peptide.fragment(
        ion_types={
            IonType.precursor: {"neutral_losses": [], "itraq": True},
            IonType.imm: {},
            IonType.b: {"neutral_losses": []},
            IonType.y: {"neutral_losses": []},
            IonType.a: {"neutral_losses": []}
        })]
