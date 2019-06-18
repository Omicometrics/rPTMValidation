#! /usr/bin/env python3
"""
Generate decoy peptides from the target protein sequence database.

"""
import argparse
import csv
import os
from typing import List, Tuple

from pepfrag import AA_MASSES, FIXED_MASSES

from . import proteolysis
from . import readers


def generate_decoy_fasta(target_db_path: str,
                         decoy_prefix: str = "_DECOY") -> str:
    """
    Generates a decoy protein sequence FASTA file from the target protein
    sequence database.

    Args:
        target_db_path (str): The path to the target protein sequence
                              database FASTA file.
        decoy_prefix (str, optional): The string with which to prepend target
                                      protein IDs.

    """
    split_path = target_db_path.rsplit(".", maxsplit=1)
    decoy_path = f"{split_path[0]}_reversed.fasta"

    if os.path.exists(decoy_path):
        print(f"Using existing decoy proteins at {decoy_path}")
        return decoy_path

    print("Generating reversed protein sequences")
    with open(decoy_path, "w") as dfh:
        with open(target_db_path) as tfh:
            for title, protein in readers.read_fasta_sequences(tfh):
                prot_id = title.split()[0][1:]
                dfh.write(f">{decoy_prefix}_{prot_id}\n")
                dfh.write(f"{protein[::-1]}\n")

    return decoy_path


def generate_decoy_file(target_db_path: str,
                        proteolyzer: proteolysis.Proteolyzer,
                        decoy_prefix: str = '_DECOY') -> str:
    """
    Generates the decoy peptide sequence database file.

    Args:
        target_db_path (str): The path to the target protein sequence
                              database.
        proteolyzer (proteolysis.Proteolyzer): The protein digester object.
        decoy_prefix (str, optional): The string with which to prepend target
                                      protein IDs.

    Returns:
        The path to the newly-generated decoy file.

    """
    split_path = target_db_path.rsplit('.', maxsplit=1)
    decoy_path = f"{split_path[0]}_reversed_{proteolyzer.enzyme}" \
                 f"_digested.csv"

    if os.path.exists(decoy_path):
        print(f"Using existing decoy peptides at {decoy_path}")
        return decoy_path

    print("Generating decoy peptides")
    with open(decoy_path, 'w', newline='') as dfh:
        writer = csv.writer(dfh, delimiter='\t')
        writer.writerow(("Protein_Name", "Sequence", "Monoisotopic_Mass"))
        with open(target_db_path) as tfh:
            rows: List[Tuple[str, str, str]] = []
            count = 0
            for title, protein in readers.read_fasta_sequences(tfh):
                peptides = proteolyzer.cleave(protein[::-1], numbermissed=2)
                if not peptides:
                    continue
                prot_id = title.split()[0][1:]
                rows += [(f">{decoy_prefix}_{prot_id}", pep,
                          "{:.6f}".format(sum(AA_MASSES[aa].mono for aa in pep)
                                          + FIXED_MASSES["H2O"]))
                         for pep in peptides]
                count += 1
                if count == 1000:
                    writer.writerows(rows)
                    rows, count = [], 0
            writer.writerows(rows)

    return decoy_path


def parse_args():
    """
    Parses the command line arguments to the script.

    Returns:
        argparse.Namespace

    """
    parser = argparse.ArgumentParser(
        description="Generate decoy peptides from fasta protein sequence"
                    "database file for peptide identification/validation")
    parser.add_argument(
        'fasta',
        metavar='*.fasta|*.fa',
        help='FASTA file of target proteins sequences for which to create '
             'decoys')
    parser.add_argument(
        '--enzyme',
        '-e',
        dest='enzyme',
        default='Trypsin',
        help='Enzyme for proteolytic cleavage. Default = Trypsin')
    parser.add_argument(
        '--decoy_prefix',
        '-d',
        dest='dprefix',
        default='DECOY_',
        help='Set accesion prefix for decoy proteins in output. '
             'Default=DECOY_')
    return parser.parse_args()


def main():
    """
    The main entry point for the CLI.

    """
    args = parse_args()
    generate_decoy_file(args.fasta, proteolysis.Proteolyzer(args.enzyme),
                        decoy_prefix=args.dprefix)


if __name__ == "__main__":
    main()
