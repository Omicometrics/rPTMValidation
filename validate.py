#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import argparse
import collections
import json

import readers
from constants import AA_MASSES, FIXED_MASSES
import utilities


SpecMatch = collections.namedtuple("SpecMatch",
                                   ["seq", "mods", "theor_z", "conf"])
                                   
Ident = collections.namedtuple(
    "Ident", ["file", "spec", "seq", "mods", "theor_z"])
    
    
def get_identifications(target_mod, target_residues, input_files,
                        unimod_ptms):
    """
    Retrieves the identification results from the set of input files.
    
    Args:
        target_mod (str): The modification for which to retrieve
                          identifications.
        target_residues (list): The residues targeted by target_mod.
        input_files (dict): A map of file paths to confidence values.
        unimod_ptms (dict): The UniMod PTM DB, read from read_unimod_ptms.
        
    Returns:
        (list, dict): The identifications for the target modification and all
                      ProteinPilot results, keyed by input file path.

    """
    idents = []
    pp_res = collections.defaultdict(lambda: collections.defaultdict(list))
    for ifile, conf in input_files.items():
        summaries = readers.read_peptide_summary(
            ifile, condition=lambda r: float(r["Conf"]) >= conf)
        for summary in summaries:
            mods = summary.mods.replace(' ', '')
            mods = ';'.join(m.split("ProteinTerminal")[1]
                            if m.startswith("ProteinTerminal") else m
                            for m in mods.split(';')
                            if not m.startswith("No"))

            try:
                utilities.parse_mods(mods, unimod_ptms)
            except utilities.UnknownModificationException:
                continue
                
            if any(f"{target_mod}({tr})" in summary.mods
                   for tr in target_residues):
                idents.append(
                    Ident(ifile, summary.spec, summary.seq,
                          mods, summary.theor_z))
                
            pp_res[ifile][summary.spec].append(
                SpecMatch(summary.seq, mods, summary.theor_z, summary.conf))
                
    return idents, pp_res


def validate(target_mod, target_residues, input_files,
             uniprot_ptm_file, unimod_ptm_file):
    """
    Processes the set of input data files to validate the PTM identifications.
    
    Args:
        target_mod (str): The modification to be validated.
        target_residues (list): The residues to be targeted by target_mod.
        input_files (dict): A map of file paths to confidence values.
        uniprot_ptm_file (str): The path to the UniProt PTM list file.
        unimod_ptm_file (str): The path to the UniMod PTM list file.
        
    """
    print(f'Processing {len(input_files)} data files')
    uniprot_ptms = readers.read_uniprot_ptms(uniprot_ptm_file)
    unimod_ptms = readers.read_unimod_ptms(unimod_ptm_file)

    idents, pp_res = get_identifications(target_mod, target_residues,
                                         input_files, unimod_ptms)
                                         
    # TODO: find an input file which actually contains nitration identifications
                                         
    print(idents)


def parse_args():
    """
    Parses the command line arguments to the script.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help=("The path to the JSON configuration file. "
              "See example_input.json for an example"))
    return parser.parse_args()

    
def main():
    args = parse_args()
    with open(args.config) as fh:
        config = json.load(fh)
    validate(config['modification'], config['target_residues'],
             config['input_files'],
             config.get('uniprot_ptm_file', 'ptmlist.txt'),
             config.get('unimod_ptm_file', 'unimod.txt'))

    
if __name__ == '__main__':
    main()