#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import argparse
import json

from rPTMDetermine import Validator, RPTMDetermineConfig


def parse_args():
    """
    Parses the command line arguments to the script.

    Returns:
        The parsed command line arguments in an argparse Namespace.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help=("The path to the JSON configuration file. "
              "See example_input.json for an example")
    )
    return parser.parse_args()


def main():
    """
    The main entry point for the rPTMDetermine validation code.

    """
    args = parse_args()
    with open(args.config) as handle:
        conf = RPTMDetermineConfig(json.load(handle))

    validator = Validator(conf)
    validator.validate()


if __name__ == '__main__':
    main()
