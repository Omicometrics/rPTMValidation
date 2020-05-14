#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import argparse
import json
import pickle

from rPTMDetermine import Validator, ValidatorConfig
from rPTMDetermine.validator import write_results


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
    """
    The main entry point for the rPTMDetermine code.

    """
    args = parse_args()
    with open(args.config) as handle:
        conf = ValidatorConfig(json.load(handle))

    validator = Validator(conf)
    validator.validate()

    with open(validator.file_prefix + "validated_psms", "wb") as fh:
        pickle.dump(validator.psms, fh)

    validator.localize()

    write_results(validator.file_prefix + "results.csv", validator.psms)

    with open(validator.file_prefix + "final_psms", "wb") as fh:
        pickle.dump(validator.psms, fh)


if __name__ == '__main__':
    main()
